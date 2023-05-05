import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
import sklearn.metrics
from sklearn.metrics import precision_recall_fscore_support

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, bias=None):
        attn = torch.matmul(q / self.scale, k.transpose(-1, -2))
        if bias is not None:
            attn += bias
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_key, n_head, dropout):
        super(MultiHeadAttention, self).__init__()
        if d_model % n_head != 0:
            raise ValueError("The hidden size is not a multiple of the number of attention heads")
        self.n_head = n_head
        self.d_k = d_key // n_head
        self.fc_query = nn.Linear(d_model, d_key, bias=False)
        self.fc_key = nn.Linear(d_model, d_key, bias=False)
        self.fc_value = nn.Linear(d_model, d_key, bias=False)
        self.attention = ScaledDotProductAttention(scale=self.d_k ** 0.5, dropout=dropout)
        self.fc_out = nn.Linear(d_key, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        """
        x has shape (*, L, C)
        return shape (*, nhead, L, C/nhead)
        """
        new_shape = x.shape[:-1] + (self.n_head, -1)
        x = x.view(*new_shape)
        return x.transpose(-3, -2)

    def forward(self, x, bias=None):
        q = self.transpose_for_scores(self.fc_query(x))
        k = self.transpose_for_scores(self.fc_key(x))
        v = self.transpose_for_scores(self.fc_value(x))
        x, attn_weight = self.attention(q, k, v, bias=bias)
        x = x.transpose(-3, -2)
        x = x.reshape(*x.shape[:-2], -1)
        x = self.dropout(self.fc_out(x))
        return x, attn_weight


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForward, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ff(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_key, n_head, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model, d_key=d_key, n_head=n_head, dropout=dropout
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)

    def forward(self, x, bias):
        branch, attn_weight = self.attn(self.norm1(x), bias)
        x = x + branch
        x = x + self.ffn(self.norm2(x))
        return x, attn_weight


class TransformerEncoder(nn.Module):
    def __init__(self, n_layer, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(**kwargs) for _ in range(n_layer)])

    def forward(self, x, bias):
        attn_weight = []
        for module in self.layers:
            x, w = module(x, bias)
            attn_weight.append(w)
        return x, attn_weight

class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=64, max_distance=256, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        num_buckets //= 2
        ret = (relative_position < 0).to(relative_position) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(relative_position / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, relative_position):
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bias = self.relative_attention_bias(rp_bucket)
        return rp_bias

class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.d_model = 256
        self.n_layer = 8

        n_heads = self.d_model // 32

        dropout = 0.1
        self.embed_tokens = nn.Sequential(
            nn.Embedding(5, self.d_model), nn.Dropout(dropout)
        )
        self.embed_rp = RelativePositionBias(
            num_buckets=64, max_distance=256, n_heads=n_heads
        )

        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            d_key=self.d_model,
            n_layer=self.n_layer,
            n_head=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
        )

        self.layer_norm_after = nn.LayerNorm(self.d_model)
        self.fc_pair_q = nn.Linear(self.d_model, self.d_model)
        self.fc_pair_k = nn.Linear(self.d_model, self.d_model)

        self.fc_pair_rp = RelativePositionBias(
            num_buckets=64, max_distance=256, n_heads=self.d_model
        )

        self.fc_pair_cls = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 2),
        )

    def forward(self, data, inference_only=False):
        """
        tokens has shape (1, L)
        """
        # 1. make embedding
        L = data["seq"].shape[1]
        x = self.embed_tokens(data["seq"])

        # 2. make attention bias
        pos = torch.arange(L, device=x.device)
        pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        rp = self.embed_rp(pos)
        rp_bias = rp.permute(2, 0, 1)
        bias = rp_bias

        # 3. encoder
        if True or not self.training:
            x, attn_weight = self.encoder(x, bias)
        else:
            x, attn_weight = torch.utils.checkpoint.checkpoint(self.encoder, x, bias)

        x = self.layer_norm_after(x)
        attn_weight = torch.cat(attn_weight, dim=1)

        # 4. pair output
        q = self.fc_pair_q(x)
        k = self.fc_pair_k(x)

        px = torch.einsum("bhc,blc->bhlc", q, k)
        rp = self.fc_pair_rp(pos)
        px = px + rp
        px = self.fc_pair_cls(px)
        out = {"contact_logits": px}
        if inference_only:
            return out
        out.update(self._compute_loss(out, data))
        if not self.training:
            with torch.no_grad():
                out.update(self._calc_metric(out, data))
        return out

    def _compute_loss(self, out, data):
        pred = out["contact_logits"].permute(0, 3, 1, 2)
        truth = data["contact"]
        weight = torch.tensor([1.0, 1.0], device=pred.device)
        return {
            "cross_entropy_loss": F.cross_entropy(
                pred, truth, weight=weight, ignore_index=-1
            )
        }

    def _calc_metric(self, out, data):
        pred = torch.softmax(out["contact_logits"].permute(0, 3, 1, 2), dim=1)
        pred = (pred + pred.transpose(-1, -2)) / 2.0
        pred = pred[:, 1]
        truth = data["contact"]

        mask = torch.triu(torch.ones_like(pred, dtype=torch.bool), 1)
        pred, truth = pred[mask], truth[mask]
        pseudoknot = data["pseudoknot"][mask]
        pred = pred.cpu().numpy()
        truth = truth.cpu().numpy()
        pseudoknot = pseudoknot.cpu().numpy()

        pred_prob = pred
        pred = pred > 0.3

        prec, recall, f1, _ = precision_recall_fscore_support(
            y_true=truth, y_pred=pred, zero_division=0, average="binary"
        )
        metric = {"precision_m": 100.0 * prec, "recall_m": 100.0 * recall, "F1_m": f1}

        for status in [0, 1, 2]:
            prec, recall, f1, _ = precision_recall_fscore_support(
                y_true=truth[pseudoknot == status],
                y_pred=pred[pseudoknot == status],
                zero_division=0,
                average="binary",
            )
            metric["precision_pseudoknot%i_m" % status] = 100.0 * prec
            metric["recall_pseudoknot%i_m" % status] = 100.0 * recall
            metric["F1_pseudoknot%i_m" % status] = f1

            metric["pred%i" % status] = pred_prob[pseudoknot == status]
            metric["truth%i" % status] = truth[pseudoknot == status]

        return metric
