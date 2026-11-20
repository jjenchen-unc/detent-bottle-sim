"""ACT (Action Chunking with Transformers, Zhao et al. 2023), compact version.

A shared ResNet-18 encodes every camera; a transformer encoder fuses image tokens with
proprioception and a CVAE style latent; a transformer decoder emits a chunk of future absolute
actions. During training a second transformer encodes the ground-truth chunk into the latent
(CVAE posterior); at test time the latent is zero.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d


def encoder(d, heads, ff, layers, dropout):
    layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)


class ACT(nn.Module):
    def __init__(self, n_cams, proprio_dim, action_dim, chunk=50, image_size=96, d=256, heads=8, enc_layers=4,
                 dec_layers=4, vae_layers=4, ff=1024, latent=32, dropout=.1, pretrained=True, feature_layer=4):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        # children: conv1 bn1 relu maxpool layer1..layer4 avgpool fc; layer3 keeps a 2x finer grid
        self.backbone = nn.Sequential(*list(resnet.children())[:4 + feature_layer])
        self.proj = nn.Conv2d(64 * 2 ** (feature_layer - 1), d, 1)
        stride = 2 ** (feature_layer + 1)
        hw = (-(-image_size // stride)) ** 2   # ResNet-18 feature map is ceil(S / stride) square
        self.cam_embed = nn.Parameter(torch.randn(n_cams, 1, d) * .02)
        self.grid_embed = nn.Parameter(torch.randn(1, hw, d) * .02)
        self.extra_embed = nn.Parameter(torch.randn(2, d) * .02)
        self.latent_proj = nn.Linear(latent, d)
        self.proprio_proj = nn.Linear(proprio_dim, d)
        self.encoder = encoder(d, heads, ff, enc_layers, dropout)
        layer = nn.TransformerDecoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, dec_layers, norm=nn.LayerNorm(d))
        self.queries = nn.Parameter(torch.randn(chunk, d) * .02)
        self.head = nn.Linear(d, action_dim)
        # CVAE posterior q(z | proprio, action chunk)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * .02)
        self.vae_action = nn.Linear(action_dim, d)
        self.vae_proprio = nn.Linear(proprio_dim, d)
        self.vae_pos = nn.Parameter(torch.randn(chunk + 2, d) * .02)
        self.vae_encoder = encoder(d, heads, ff, vae_layers, dropout)
        self.to_latent = nn.Linear(d, 2 * latent)
        self.latent = latent

    def forward(self, images, proprio, actions=None, mask=None):
        """images (B, C, 3, H, W) ImageNet-normalised; proprio (B, P); actions (B, chunk, A) and
        mask (B, chunk) valid steps, both only for training. Returns (pred, mu, logvar)."""
        B, C = images.shape[:2]
        if actions is not None:
            tokens = torch.cat([self.cls.expand(B, -1, -1), self.vae_proprio(proprio)[:, None], self.vae_action(actions)], 1)
            pad = torch.cat([torch.zeros(B, 2, dtype=torch.bool, device=mask.device), ~mask], 1)
            h = self.vae_encoder(tokens + self.vae_pos, src_key_padding_mask=pad)[:, 0]
            mu, logvar = self.to_latent(h).chunk(2, -1)
            z = mu + torch.randn_like(mu) * (.5 * logvar).exp()
        else:
            mu = logvar = None
            z = proprio.new_zeros(B, self.latent)
        feats = self.proj(self.backbone(images.flatten(0, 1)))                 # (B*C, d, h, w)
        feats = feats.flatten(2).transpose(1, 2).reshape(B, C, -1, feats.shape[1])
        feats = (feats + self.cam_embed + self.grid_embed).flatten(1, 2)       # (B, C*hw, d)
        extra = torch.stack([self.latent_proj(z), self.proprio_proj(proprio)], 1) + self.extra_embed
        memory = self.encoder(torch.cat([extra, feats], 1))
        out = self.decoder(self.queries.expand(B, -1, -1), memory)
        return self.head(out), mu, logvar


def act_loss(pred, target, mask, mu, logvar, kl_weight):
    l1 = (F.l1_loss(pred, target, reduction='none').mean(-1) * mask).sum() / mask.sum()
    kl = (-.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1)).mean()
    return l1 + kl_weight * kl, dict(l1=l1.item(), kl=kl.item())
