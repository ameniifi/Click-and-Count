import argparse
import json
import os
import re
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import open_clip
except ImportError:  
    open_clip = None


TEXT_KEY_GROUPS = (
    (
        'descriptions', 'description', 'text_descriptions', 'text_description',
        'captions', 'caption',
        'texts', 'text', 'prompts', 'prompt', 'sentences', 'sentence'
    ),
    (
        'class_names', 'class_name', 'classes', 'class',
        'labels', 'label', 'category', 'categories'
    ),
)

IGNORED_KEYS = {
    'image', 'image_name', 'img', 'img_name', 'filename', 'file_name', 'fname',
    'split', 'subset', 'id', 'idx', 'index', 'ann_id', 'bbox', 'bboxes', 'box',
    'boxes', 'point', 'points', 'count', 'counts', 'density_map', 'metadata',
}

IMAGE_NAME_RE = re.compile(r'.*\.(jpg|jpeg|png|bmp|gif|tif|tiff|webp)$', re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser("Precompute text embeddings")
    parser.add_argument('--input_json', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--model_name', default='ViT-B-16', type=str)
    parser.add_argument('--pretrained', default='openai', type=str)
    parser.add_argument('--image_key', default='image_name', type=str)
    parser.add_argument('--text_key', default='text', type=str)
    parser.add_argument('--prompt_template', default='', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str
    )
    return parser.parse_args()


def load_text_records(input_json, image_key, text_key, prompt_template):
    with open(input_json, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)

    records = list()

    def normalize_prompt(raw_prompt):
        prompt = str(raw_prompt)
        if prompt_template:
            prompt = prompt_template.format(prompt)
        return prompt

    def is_usable_text(text):
        if not isinstance(text, str):
            return False
        prompt = text.strip()
        if not prompt:
            return False
        if IMAGE_NAME_RE.match(prompt):
            return False
        if prompt.isdigit():
            return False
        return any(char.isalpha() for char in prompt)

    def deduplicate(prompts):
        deduplicated = OrderedDict()
        for prompt in prompts:
            prompt = normalize_prompt(prompt)
            if is_usable_text(prompt):
                deduplicated[prompt] = None
        return list(deduplicated.keys())

    def extract_prompts(value):
        if isinstance(value, str):
            return [value.strip()] if is_usable_text(value) else []
        if isinstance(value, bytes):
            return extract_prompts(value.decode('utf-8', errors='ignore'))
        if isinstance(value, (list, tuple)):
            prompts = list()
            for item in value:
                prompts.extend(extract_prompts(item))
            return deduplicate(prompts)
        if isinstance(value, dict):
            prompts = list()

            # Prefer fields that are most likely to contain human-readable text.
            for group in TEXT_KEY_GROUPS:
                for key in group:
                    if key in value:
                        prompts.extend(extract_prompts(value[key]))
                if prompts:
                    return deduplicate(prompts)

            # Fall back to recursively scanning the remaining fields.
            for key, item in value.items():
                if str(key).lower() in IGNORED_KEYS:
                    continue
                prompts.extend(extract_prompts(item))
            return deduplicate(prompts)
        if isinstance(value, np.ndarray):
            return extract_prompts(value.tolist())
        return list()

    if isinstance(payload, dict):
        for image_name, value in payload.items():
            if image_name == 'metadata':
                continue
            prompts = extract_prompts(value)
            if not prompts and isinstance(value, dict) and text_key in value:
                prompts = extract_prompts(value[text_key])
            if not prompts:
                available_keys = ', '.join(sorted(str(key) for key in value.keys())) if isinstance(value, dict) else type(value).__name__
                raise ValueError(
                    f"Could not find a usable text field for {image_name} in "
                    f"{input_json}. Available keys: {available_keys}"
                )
            records.append((str(image_name), prompts))
    elif isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"List entries in {input_json} must be JSON objects."
                )
            image_name = (
                entry.get(image_key)
                or entry.get('image_name')
                or entry.get('image')
                or entry.get('img')
                or entry.get('file_name')
                or entry.get('filename')
                or entry.get('fname')
            )
            if image_name is None:
                raise ValueError(
                    f"Missing image key '{image_key}' in one entry of {input_json}."
                )
            prompt_source = dict(entry)
            for key in {
                image_key, 'image_name', 'image', 'img', 'file_name', 'filename', 'fname'
            }:
                prompt_source.pop(key, None)
            prompts = extract_prompts(prompt_source)
            if not prompts and text_key in entry:
                prompts = extract_prompts(entry[text_key])
            if not prompts:
                raise ValueError(
                    f"Could not find a usable text field for {image_name} in {input_json}."
                )
            records.append((str(image_name), prompts))
    else:
        raise ValueError(
            f"Unsupported JSON format in {input_json}; expected a dict or a list."
        )

    deduplicated = OrderedDict()
    for image_name, prompts in records:
        deduplicated[image_name] = prompts

    return list(deduplicated.items())


def encode_text_records(records, model_name, pretrained, batch_size, device):
    if open_clip is None:
        raise ImportError(
            "open_clip_torch is required to build the cached text embeddings."
        )

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    image_names = list()
    prompts = list()
    text_tokens = list()
    text_features = list()

    for start in tqdm(range(0, len(records), batch_size), desc='Encoding text'):
        batch = records[start:start + batch_size]
        batch_names = list()
        batch_prompt_groups = list()
        batch_prompts = list()
        prompt_counts = list()

        for image_name, prompt_group in batch:
            prompt_group = [prompt for prompt in prompt_group if prompt]
            if not prompt_group:
                raise ValueError(f"No prompts found for {image_name}.")
            batch_names.append(image_name)
            batch_prompt_groups.append(prompt_group)
            batch_prompts.extend(prompt_group)
            prompt_counts.append(len(prompt_group))

        tokens = tokenizer(batch_prompts).to(device)
        with torch.no_grad():
            text_output = model.forward_intermediates(
                text=tokens,
                text_indices=1,
                normalize=True,
                normalize_intermediates=True,
                intermediates_only=False,
            )

            if 'text_intermediates' not in text_output or not text_output['text_intermediates']:
                raise RuntimeError(
                    "open_clip did not return text_intermediates. "
                    "Install a recent open_clip_torch release with forward_intermediates support."
                )

            batch_text_tokens = text_output['text_intermediates'][-1].float()
            batch_text_features = text_output['text_features'].float()

        offset = 0
        batch_tokens = list()
        batch_features = list()
        for prompt_count in prompt_counts:
            prompt_tokens = batch_text_tokens[offset:offset + prompt_count]
            prompt_features = batch_text_features[offset:offset + prompt_count]
            offset += prompt_count
            pooled_tokens = prompt_tokens.mean(dim=0)
            pooled_features = F.normalize(prompt_features.mean(dim=0), dim=-1)
            batch_tokens.append(pooled_tokens.cpu())
            batch_features.append(pooled_features.cpu())

        image_names.extend(batch_names)
        prompts.extend(batch_prompt_groups)
        text_tokens.append(torch.stack(batch_tokens, dim=0))
        text_features.append(torch.stack(batch_features, dim=0))

    return image_names, prompts, torch.cat(text_tokens, dim=0), torch.cat(text_features, dim=0)


def main():
    args = parse_args()
    records = load_text_records(
        args.input_json, args.image_key, args.text_key, args.prompt_template
    )

    if not records:
        raise ValueError(f"No text records were found in {args.input_json}.")

    image_names, prompts, text_tokens, text_features = encode_text_records(
        records, args.model_name, args.pretrained, args.batch_size, args.device
    )

    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    torch.save(
        {
            'image_names': image_names,
            'prompts': prompts,
            'text_tokens': text_tokens.contiguous(),
            'text_features': text_features.contiguous(),
            'metadata': {
                'input_json': args.input_json,
                'model_name': args.model_name,
                'pretrained': args.pretrained,
                'prompt_template': args.prompt_template,
                'token_dim': text_tokens.size(-1),
                'feature_dim': text_features.size(-1),
            },
        },
        args.output_path
    )

    print(
        f"Saved {len(image_names)} text embeddings to {args.output_path} "
        f"with token dim {text_tokens.size(-1)} and feature dim {text_features.size(-1)}."
    )


if __name__ == '__main__':
    main()
