import os
from pathlib import Path

import numpy as np
import torch


def _as_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).detach().cpu().float()
    return torch.as_tensor(value, dtype=torch.float32).detach().cpu().float()


def coerce_text_vector(value):
    tensor = _as_tensor(value)

    if tensor.numel() == 0:
        return tensor.reshape(-1)

    while tensor.dim() > 1:
        tensor = tensor.mean(dim=0)

    return tensor.flatten().contiguous()


def clone_text_embedding_bundle(bundle):
    if isinstance(bundle, dict):
        return {
            key: clone_text_embedding_bundle(value)
            for key, value in bundle.items()
        }
    if isinstance(bundle, (list, tuple)):
        return type(bundle)(clone_text_embedding_bundle(value) for value in bundle)
    return _as_tensor(bundle).clone().contiguous()


def annotate_text_embedding_bundle(bundle, present: bool):
    cloned = clone_text_embedding_bundle(bundle)

    if isinstance(cloned, dict):
        if not present:
            for key in ('tokens', 'text_tokens'):
                if key in cloned and isinstance(cloned[key], torch.Tensor):
                    cloned[key] = torch.zeros_like(cloned[key])
            for key in ('features', 'text_features'):
                if key in cloned and isinstance(cloned[key], torch.Tensor):
                    cloned[key] = torch.zeros_like(cloned[key])
        cloned['present'] = torch.tensor(float(bool(present)), dtype=torch.float32)
        return cloned

    if isinstance(cloned, torch.Tensor):
        return torch.zeros_like(cloned) if not present else cloned

    return cloned


def move_text_embedding_bundle(bundle, device):
    if bundle is None:
        return None
    if isinstance(bundle, dict):
        return {
            key: move_text_embedding_bundle(value, device)
            for key, value in bundle.items()
        }
    if isinstance(bundle, (list, tuple)):
        return type(bundle)(move_text_embedding_bundle(value, device) for value in bundle)
    return _as_tensor(bundle).to(device=device)


def batch_text_embedding_bundles(bundle_list):
    """
    Converts a list of per-sample text bundles into a single batched bundle.

    Training-time text dropout is represented as zeroed bundles with a
    `present` flag, so only truly empty tensors disable text for the batch.
    """
    if not bundle_list:
        return None

    for bundle in bundle_list:
        if isinstance(bundle, torch.Tensor) and bundle.numel() == 0:
            return None

    if all(isinstance(bundle, dict) for bundle in bundle_list):
        batched = {}
        for key in bundle_list[0].keys():
            values = [bundle[key] for bundle in bundle_list]
            try:
                batched[key] = torch.stack(values, dim=0)
            except RuntimeError:
                batched[key] = values
        return batched

    if all(isinstance(bundle, torch.Tensor) for bundle in bundle_list):
        try:
            return torch.stack(bundle_list, dim=0)
        except RuntimeError:
            return bundle_list

    return bundle_list


def split_text_embedding_bundle(bundle):
    if bundle is None:
        return None, None

    if isinstance(bundle, dict):
        tokens = bundle.get('tokens', bundle.get('text_tokens'))
        features = bundle.get('features', bundle.get('text_features'))

        if tokens is None and features is None:
            if len(bundle) == 1:
                return split_text_embedding_bundle(next(iter(bundle.values())))
            return None, None

        if tokens is not None:
            tokens = _as_tensor(tokens)
        if features is not None:
            features = _as_tensor(features)

        if tokens is None and features is not None:
            return None, features
        if tokens is not None and features is None:
            if tokens.dim() == 2:
                return None, tokens
            if tokens.dim() >= 3:
                return tokens, tokens.mean(dim=1)
            return None, tokens
        return tokens, features

    if isinstance(bundle, (list, tuple)):
        if len(bundle) == 0:
            return None, None
        if len(bundle) == 1:
            return split_text_embedding_bundle(bundle[0])
        if len(bundle) == 2:
            tokens = None if bundle[0] is None else _as_tensor(bundle[0])
            features = None if bundle[1] is None else _as_tensor(bundle[1])
            return tokens, features

    tensor = _as_tensor(bundle)
    if tensor.numel() == 0:
        return None, None
    if tensor.dim() == 1:
        return None, tensor
    if tensor.dim() == 2:
        return None, tensor
    if tensor.dim() >= 3:
        return tensor, tensor.mean(dim=1)
    return None, tensor


def text_embedding_presence(bundle):
    if bundle is None:
        return None

    if isinstance(bundle, dict):
        for key in ('present', 'text_present', 'valid', 'mask', 'valid_mask'):
            if key in bundle:
                return _as_tensor(bundle[key])

    return None


def _text_feature_dim(bundle):
    tokens, features = split_text_embedding_bundle(bundle)
    reference = features if features is not None else tokens
    if reference is None:
        return 0
    return reference.shape[-1]


def _load_embedding_object(path):
    suffix = path.suffix.lower()
    if suffix in {'.pt', '.pth', '.bin'}:
        return torch.load(path, map_location='cpu')
    if suffix == '.npy':
        return np.load(path, allow_pickle=True)
    if suffix == '.npz':
        data = np.load(path, allow_pickle=True)
        if len(data.files) == 1:
            return data[data.files[0]]
        if 'embedding' in data.files:
            return data['embedding']
        if 'embeddings' in data.files:
            return data['embeddings']
        return data
    raise ValueError(f"Unsupported text embedding file type: {path}")


def _coerce_loaded_item(value):
    if isinstance(value, dict):
        if 'tokens' in value or 'text_tokens' in value or 'features' in value or 'text_features' in value:
            text_bundle = dict()
            tokens = value.get('tokens', value.get('text_tokens'))
            features = value.get('features', value.get('text_features'))
            if tokens is not None:
                text_bundle['tokens'] = _as_tensor(tokens)
            if features is not None:
                text_bundle['features'] = _as_tensor(features)
            return text_bundle
        for key in ('embedding', 'embeddings', 'text_embedding', 'vector', 'cls'):
            if key in value:
                return coerce_text_vector(value[key])
        if len(value) == 1:
            return coerce_text_vector(next(iter(value.values())))
        raise ValueError("Could not find an embedding tensor inside the loaded object.")
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            tokens, features = value
            if isinstance(tokens, (torch.Tensor, np.ndarray, list, tuple)) or isinstance(
                features, (torch.Tensor, np.ndarray, list, tuple)
            ):
                return {
                    'tokens': _as_tensor(tokens),
                    'features': _as_tensor(features),
                }
        return coerce_text_vector(value)
    return coerce_text_vector(value)


def _candidate_keys(path):
    keys = [path.name, path.stem, path.with_suffix('').name]
    return list(dict.fromkeys(key for key in keys if key))


def _scan_embedding_directory(directory_path):
    supported_suffixes = {'.pt', '.pth', '.bin', '.npy', '.npz'}
    text_cache = dict()

    for file_path in sorted(directory_path.rglob('*')):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_suffixes:
            continue

        embedding = _coerce_loaded_item(_load_embedding_object(file_path))
        for key in _candidate_keys(file_path):
            text_cache[key] = clone_text_embedding_bundle(embedding)

    if not text_cache:
        raise ValueError(f"No supported embedding files were found in {directory_path}")

    return text_cache


def _torch_load_cpu(path):
    """
    Wraps torch.load(map_location='cpu') and tries mmap=True first.

    mmap=True (torch>=2.1) memory-maps the file instead of reading the
    whole blob into RAM up front, which matters for large COCO-scale text
    embedding caches (~20GB) on memory-constrained SLURM allocations.
    Falls back to a plain load on older torch versions that don't support
    the mmap kwarg.
    """
    try:
        return torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        # older torch without the mmap kwarg
        return torch.load(path, map_location='cpu')


def infer_text_embedding_dim(text_embedding_path):
    if not text_embedding_path:
        return 0

    path = Path(text_embedding_path)
    if path.is_dir():
        cache = _scan_embedding_directory(path)
        return _text_feature_dim(next(iter(cache.values())))

    cache = load_text_embedding_cache(text_embedding_path)
    if not cache:
        return 0
    return _text_feature_dim(next(iter(cache.values())))


def infer_text_embedding_dim_from_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return 0

    candidate_suffixes = (
        'projector.proj.weight',
        'gamma.weight',
        'beta.weight',
        'text_projection.weight',
        'text_proj.weight',
    )

    for key, value in state_dict.items():
        if 'text_conditioners' not in key:
            continue
        if not isinstance(value, torch.Tensor):
            continue
        if any(key.endswith(suffix) for suffix in candidate_suffixes):
            return value.shape[-1] if value.dim() >= 2 else value.numel()

    return 0


def load_text_embedding_keys(text_embedding_path):
    """
    Returns the set of available sample keys in a text embedding cache
    WITHOUT materializing tensor data.

    Used by verify_coco_counting.py, which only needs key-existence checks
    (`sample_key not in text_cache`) and never touches embedding values.
    load_text_embedding_cache() clones every per-image tensor into a new
    dict (see the structured-cache branch below), which roughly doubles
    peak memory on top of the torch.load() blob itself -- for COCO-scale
    token caches (~20GB for ~123K images at [77, 512] float32) this OOMs
    a 16G SLURM allocation. This function skips that clone step entirely,
    and additionally tries mmap=True to avoid even the full-file read
    where the torch version supports it.
    """
    if not text_embedding_path:
        return None

    path = Path(text_embedding_path)
    if path.is_dir():
        supported_suffixes = {'.pt', '.pth', '.bin', '.npy', '.npz'}
        keys = set()
        for file_path in sorted(path.rglob('*')):
            if not file_path.is_file() or file_path.suffix.lower() not in supported_suffixes:
                continue
            keys.update(_candidate_keys(file_path))
        if not keys:
            raise ValueError(f"No supported embedding files were found in {path}")
        return keys

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing text embedding cache: {text_embedding_path}"
        )

    cache = _torch_load_cpu(path)

    if isinstance(cache, dict) and 'image_names' in cache:
        return set(str(name) for name in cache['image_names'])

    if isinstance(cache, dict):
        return set(str(k) for k in cache.keys() if k != 'metadata')

    raise ValueError(
        f"Unsupported text embedding cache format in {text_embedding_path}"
    )


def load_text_embedding_cache(text_embedding_path):
    if not text_embedding_path:
        return None

    path = Path(text_embedding_path)
    if path.is_dir():
        return _scan_embedding_directory(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing text embedding cache: {text_embedding_path}"
        )

    # mmap=True (torch>=2.1) avoids reading the whole file into RAM up
    # front where supported; falls back to a plain load otherwise. This
    # only reduces the initial read -- the clone()/dict-rebuild below
    # still materializes full tensor data, which is required here since
    # callers of this function (e.g. training) actually use the values.
    # Callers that only need key existence should use
    # load_text_embedding_keys() instead.
    cache = _torch_load_cpu(path)

    if isinstance(cache, dict) and 'image_names' in cache:
        image_names = list(cache['image_names'])
        if 'text_tokens' in cache or 'text_features' in cache:
            tokens = cache.get('text_tokens', cache.get('tokens'))
            features = cache.get('text_features', cache.get('features'))

            if tokens is None:
                raise ValueError(
                    "Structured text cache with tokens/features must include text_tokens."
                )

            tokens = _as_tensor(tokens)
            if features is None:
                features = tokens.mean(dim=1) if tokens.dim() >= 3 else tokens
            features = _as_tensor(features)

            if len(image_names) != tokens.size(0) or len(image_names) != features.size(0):
                raise ValueError(
                    "text cache image_names, text_tokens and text_features have mismatched lengths."
                )

            return {
                str(image_name): {
                    'tokens': tokens[idx].clone().contiguous(),
                    'features': features[idx].clone().contiguous()
                }
                for idx, image_name in enumerate(image_names)
            }

        if 'embeddings' in cache:
            embeddings = cache['embeddings']
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.as_tensor(embeddings, dtype=torch.float32)

            embeddings = embeddings.detach().cpu().float()
            if len(image_names) != embeddings.size(0):
                raise ValueError(
                    "text cache image_names and embeddings have mismatched lengths."
                )

            if embeddings.dim() == 2:
                return {
                    str(image_name): embeddings[idx].clone().contiguous()
                    for idx, image_name in enumerate(image_names)
                }

            if embeddings.dim() >= 3:
                return {
                    str(image_name): {
                        'tokens': embeddings[idx].clone().contiguous(),
                        'features': embeddings[idx].mean(dim=0).clone().contiguous()
                    }
                    for idx, image_name in enumerate(image_names)
                }

            raise ValueError(
                "Unsupported structured embeddings shape in text cache."
            )

        raise ValueError(
            "Structured text cache must include text_tokens/text_features or embeddings."
        )

    if isinstance(cache, dict):
        text_cache = dict()
        for image_name, value in cache.items():
            if image_name == 'metadata':
                continue
            text_cache[str(image_name)] = _coerce_loaded_item(value)

        if not text_cache:
            raise ValueError(
                f"No usable text embeddings were found in {text_embedding_path}"
            )

        return text_cache

    raise ValueError(
        f"Unsupported text embedding cache format in {text_embedding_path}"
    )