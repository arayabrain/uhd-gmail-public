import hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from termcolor import cprint

from uhd_eeg.models.Transformer.cBraMod.criss_cross_transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)

# Authors' public release used for manuscript fine-tuning
# (https://huggingface.co/weighting666/CBraMod/blob/main/pretrained_weights.pth).
EXPECTED_BACKBONE_SHA256 = (
    "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def warn_if_backbone_hash_mismatch(path: Path | str) -> None:
    """Warn when the local backbone file does not match the manuscript release."""
    weight_path = Path(path)
    if not weight_path.is_file():
        return
    digest = _sha256_file(weight_path)
    if digest != EXPECTED_BACKBONE_SHA256:
        cprint(
            f"Warning: SHA-256 of {weight_path} is {digest}; "
            f"expected {EXPECTED_BACKBONE_SHA256} "
            "(authors' public pretrained_weights.pth used in the manuscript).",
            "yellow",
        )


class CBraModClassifier(nn.Module):
    def __init__(self, param):
        super(CBraModClassifier, self).__init__()
        self.backbone = CBraMod(
            in_dim=param.in_dim,
            out_dim=param.out_dim,
            d_model=param.d_model,
            dim_feedforward=param.dim_feedforward,
            seq_len=param.seq_len,
            n_layer=param.n_layer,
            nhead=param.nhead,
            patch_mode=param.patch_mode,
            hopsize=param.hopsize,
        )
        if param.use_backbone_weights:
            weight_path = Path(param.backbone_weight_path)
            warn_if_backbone_hash_mismatch(weight_path)
            self.backbone.load_state_dict(
                torch.load(weight_path, map_location="cpu")
            )
            cprint(
                f"Loaded backbone weights from {weight_path}",
                "green",
            )

        self.backbone.proj_out = nn.Identity()
        flattened_dim = (
            param.num_channels * 1 * param.in_dim
            if param.patch_mode == "non_overlap"
            else param.num_channels * 2 * param.in_dim
        )
        ch_dim = param.num_channels * param.in_dim
        if param.classifier == "channel_avg_twolayer":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b c d s"),
                nn.AdaptiveAvgPool2d((200, 1)),
                nn.Flatten(),
                nn.LayerNorm(ch_dim),
                nn.Linear(ch_dim, 128),
                nn.GELU(),
                nn.Dropout(param.dropout),
                nn.Linear(128, param.num_of_classes),
            )
        elif param.classifier == "avgpooling_patch_reps":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b d c s"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == "all_patch_reps_onelayer":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flattened_dim, param.num_of_classes),
            )
        elif param.classifier == "all_patch_reps_twolayer":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.LayerNorm(flattened_dim),
                nn.Linear(flattened_dim, 512),
                nn.GELU(),
                nn.Dropout(param.dropout),
                nn.Linear(512, param.num_of_classes),
            )
        elif param.classifier == "all_patch_reps":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.LayerNorm(flattened_dim),
                nn.Linear(flattened_dim, 1024),
                nn.GELU(),
                nn.Dropout(param.dropout),
                nn.Linear(1024, 256),
                nn.GELU(),
                nn.Dropout(param.dropout),
                nn.Linear(256, param.num_of_classes),
            )
        elif param.classifier == "all_patch_reps_legacy":
            self.classifier = nn.Sequential(
                Rearrange("b c s d -> b (c s d)"),
                nn.Linear(flattened_dim, 3 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(3 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.classifier(feats)
        return out


class CBraMod(nn.Module):
    def __init__(
        self,
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
        patch_mode="overlap",
        hopsize=50,
    ):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            in_dim,
            out_dim,
            d_model,
            seq_len,
            patch_mode=patch_mode,
            hopsize=hopsize,
        )
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )
        self.encoder = TransformerEncoder(
            encoder_layer, num_layers=n_layer, enable_nested_tensor=False
        )
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, x, mask=None):
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)
        out = self.proj_out(feats)
        return out


class PatchEmbedding(nn.Module):
    def __init__(
        self, in_dim, out_dim, d_model, seq_len, patch_mode="non_overlap", hopsize=50
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.patch_mode = patch_mode
        self.hopsize = hopsize

        self.positional_encoding = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=(19, 7),
                stride=(1, 1),
                padding=(9, 3),
                groups=d_model,
            ),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(1, 49),
                stride=(1, 25),
                padding=(0, 24),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 3),
                stride=(1, 1),
                padding=(0, 1),
            ),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
            num_frames = x.size(2)
            num_patch = x.size(2) // self.in_dim
            if num_frames % self.in_dim != 0:
                if self.patch_mode == "non_overlap":
                    x = x[:, :, : int(num_patch * self.in_dim)]
                    x = x.reshape(x.size(0), x.size(1), num_patch, self.in_dim)
                elif self.patch_mode == "overlap":
                    remove_samples = (num_frames - self.in_dim) % self.hopsize
                    x = x[:, :, :-remove_samples] if remove_samples != 0 else x
                    x = x.unfold(dimension=2, size=self.in_dim, step=self.hopsize)
                else:
                    raise ValueError(f"Invalid patch_mode: {self.patch_mode}")

        bz, ch_num, patch_num, patch_size = x.shape
        if mask is None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding
        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(
            bz, ch_num, patch_num, self.d_model
        )

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm="forward")
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
