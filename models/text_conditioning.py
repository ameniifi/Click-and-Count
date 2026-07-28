import torch
from torch import nn
from torch.nn import functional as F

from utils.text_embeddings import (
    split_text_embedding_bundle,
    text_embedding_presence,
)


def prepare_text_embeddings(text_embeddings):
    tokens, features = split_text_embedding_bundle(text_embeddings)

    if features is not None:
        if features.dim() == 1:
            features = features.unsqueeze(0)
        return features.float()

    if tokens is None:
        return None

    if tokens.dim() == 2:
        return tokens.mean(dim=0, keepdim=True).float()

    if tokens.dim() >= 3:
        return tokens.mean(dim=1).float()

    return None


def prepare_text_token_context(text_embeddings, device, dtype, batch_size=None):
    text_tokens, text_features = split_text_embedding_bundle(text_embeddings)

    if text_tokens is None:
        text_tokens = text_features
        if text_tokens is None:
            return None, None
        if text_tokens.dim() == 1:
            text_tokens = text_tokens.view(1, 1, -1)
        elif text_tokens.dim() == 2:
            if batch_size is not None and text_tokens.size(0) != batch_size:
                text_tokens = text_tokens.unsqueeze(0)
            else:
                text_tokens = text_tokens.unsqueeze(1)
        elif text_tokens.dim() != 3:
            raise ValueError(
                "Text features must have shape [dim], [batch, dim], or "
                "[batch, tokens, dim]."
            )
    else:
        if text_tokens.dim() == 2:
            text_tokens = text_tokens.unsqueeze(0)
        elif text_tokens.dim() != 3:
            raise ValueError(
                "Text tokens must have shape [batch, tokens, dim] or [tokens, dim]."
            )

    presence = text_embedding_presence(text_embeddings)
    if presence is not None:
        presence = presence.to(device=device, dtype=dtype).reshape(
            -1, *([1] * (text_tokens.dim() - 1))
        )
        if not torch.any(presence > 0.5):
            return None, None

    return text_tokens.to(device=device, dtype=dtype), presence


class TextContextProjector(nn.Module):
    def __init__(self, text_dim: int, emb_dim: int):
        super().__init__()

        if text_dim <= 0:
            raise ValueError("text_dim must be positive when text conditioning is enabled.")

        self.proj = nn.Linear(text_dim, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, text_embeddings):
        if text_embeddings is None:
            return None
        if text_embeddings.dim() not in (2, 3):
            raise ValueError(
                "Text embeddings must have shape [batch, dim] or [batch, tokens, dim]."
            )
        return self.norm(self.proj(text_embeddings))


class GatedTextCrossAttention(nn.Module):
    """
    Residual cross-attention from ClickCount prototypes to frozen text context.
    The text encoder stays frozen; only the projection and gate are learned.
    """

    def __init__(
        self,
        text_dim: int,
        emb_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.projector = TextContextProjector(text_dim, emb_dim)
        self.query_norm = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, text_embeddings):
        text_tokens, presence = prepare_text_token_context(
            text_embeddings, x.device, x.dtype, batch_size=x.size(1)
        )
        if text_tokens is None:
            return x

        text_tokens = self.projector(text_tokens)
        if presence is not None:
            text_tokens = text_tokens * presence
        text_tokens = text_tokens.transpose(0, 1)

        query = self.query_norm(x)
        attn_out, _ = self.attn(
            query=query,
            key=text_tokens,
            value=text_tokens,
            need_weights=False,
        )
        return x + torch.tanh(self.gate) * attn_out


class TextPrototypeInitializer(nn.Module):
    """
    Adds a frozen-text semantic prior directly to ClickCount's OPE inputs.

    The gates start at zero so a visual checkpoint loads as the original
    click/SAM ClickCount model, then fine-tuning learns how much text should steer
    the shape seed and exemplar appearance query.
    """

    def __init__(
        self,
        text_dim: int,
        emb_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.projector = TextContextProjector(text_dim, emb_dim)
        self.shape_norm = nn.LayerNorm(emb_dim)
        self.appearance_norm = nn.LayerNorm(emb_dim)
        self.shape_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout)
        self.appearance_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout
        )
        self.shape_gate = nn.Parameter(torch.zeros(1))
        self.appearance_gate = nn.Parameter(torch.zeros(1))

    def _condition_tokens(self, tokens, norm, attn, gate, text_context):
        query = norm(tokens)
        attn_out, _ = attn(
            query=query,
            key=text_context,
            value=text_context,
            need_weights=False,
        )
        return tokens + torch.tanh(gate) * attn_out

    def forward(self, shape_tokens, appearance_tokens, text_embeddings):
        text_tokens, presence = prepare_text_token_context(
            text_embeddings,
            shape_tokens.device,
            shape_tokens.dtype,
            batch_size=shape_tokens.size(1),
        )
        if text_tokens is None:
            return shape_tokens, appearance_tokens

        text_tokens = self.projector(text_tokens)
        if presence is not None:
            text_tokens = text_tokens * presence
        text_context = text_tokens.transpose(0, 1)

        shape_tokens = self._condition_tokens(
            shape_tokens,
            self.shape_norm,
            self.shape_attn,
            self.shape_gate,
            text_context,
        )
        if appearance_tokens is not None:
            appearance_tokens = self._condition_tokens(
                appearance_tokens,
                self.appearance_norm,
                self.appearance_attn,
                self.appearance_gate,
                text_context,
            )

        return shape_tokens, appearance_tokens


def text_alignment_loss(projector, prototypes, text_embeddings):
    text_tokens, text_features = split_text_embedding_bundle(text_embeddings)
    if text_tokens is None and text_features is None:
        return prototypes.new_zeros(())

    presence = text_embedding_presence(text_embeddings)

    if text_features is None:
        if text_tokens.dim() >= 3:
            text_features = text_tokens.mean(dim=1)
        elif text_tokens.dim() == 2:
            text_features = text_tokens.mean(dim=0, keepdim=True)
        else:
            text_features = text_tokens

    if text_features.dim() == 1:
        text_features = text_features.unsqueeze(0)

    text_summary = projector(text_features.to(device=prototypes.device, dtype=prototypes.dtype))

    if presence is not None:
        valid_mask = presence.to(device=prototypes.device).reshape(-1) > 0.5
        if not torch.any(valid_mask):
            return prototypes.new_zeros(())
        prototypes = prototypes[:, valid_mask, :]
        text_summary = text_summary[valid_mask]

    # Let the text summary weight prototype slots instead of collapsing the
    # whole prototype stack to a plain mean before computing the alignment loss.
    proto_tokens = prototypes.permute(1, 0, 2)
    proto_tokens = F.normalize(proto_tokens, dim=-1)
    text_query = F.normalize(text_summary, dim=-1).unsqueeze(1)
    # The vectors are already cosine-normalized, so an extra sqrt(d) scale would
    # flatten the weights and drift back toward a mean-pooled summary.
    attn_logits = (proto_tokens * text_query).sum(dim=-1)
    attn_weights = F.softmax(attn_logits, dim=1).unsqueeze(-1)
    proto_summary = (attn_weights * prototypes.permute(1, 0, 2)).sum(dim=1)

    proto_summary = F.normalize(proto_summary, dim=-1)
    text_summary = F.normalize(text_summary, dim=-1)
    return 1.0 - F.cosine_similarity(proto_summary, text_summary, dim=-1).mean()


class TextFiLM(nn.Module):
    """
    Lightweight residual FiLM block kept for compatibility with the older text branch.
    """

    def __init__(self, text_dim: int, emb_dim: int):
        super().__init__()

        if text_dim <= 0:
            raise ValueError("text_dim must be positive when text conditioning is enabled.")

        self.text_norm = nn.LayerNorm(text_dim)
        self.gamma = nn.Linear(text_dim, emb_dim)
        self.beta = nn.Linear(text_dim, emb_dim)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, text_embeddings):
        text_embeddings = prepare_text_embeddings(text_embeddings)
        if text_embeddings is None:
            return x

        text_embeddings = text_embeddings.to(device=x.device, dtype=x.dtype)

        if text_embeddings.size(0) != x.size(1):
            raise ValueError(
                "Batch size mismatch between text embeddings and prototype queries."
            )

        text_embeddings = self.text_norm(text_embeddings)
        gamma = self.gamma(text_embeddings).unsqueeze(0)
        beta = self.beta(text_embeddings).unsqueeze(0)
        return x * (1.0 + gamma) + beta
