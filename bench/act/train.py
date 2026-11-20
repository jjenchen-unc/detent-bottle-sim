"""Train ACT on bench demonstrations.

python -m bench.act.train --data data/expert_v1 --out runs/act/v1 --steps 20000
python -m bench.act.train --data data/expert_v3_wide data/expert_v3_wide_noise --out runs/act/v3 --steps 30000
"""
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from .data import Demos, split_episodes
from .model import ACT, act_loss

IMAGENET = (torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1), torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1))


def prepare_images(images, device, shift=0):
    """uint8 (B, C, H, W, 3) -> ImageNet-normalised float (B, C, 3, H, W), optional random-shift augmentation."""
    x = torch.as_tensor(images, device=device).permute(0, 1, 4, 2, 3).float() / 255
    if shift:
        B, C, _, H, W = x.shape
        flat = F.pad(x.flatten(0, 1), (shift,) * 4, mode='replicate')
        # DrQ-style random shift: integer translation of up to `shift` pixels per image
        eps = 1. / (H + 2 * shift)
        base = torch.linspace(-1 + eps, 1 - eps, H + 2 * shift, device=device)[:H]
        grid = torch.stack(torch.meshgrid(base, base, indexing='xy'), -1)[None].expand(B * C, -1, -1, -1)
        offset = torch.randint(0, 2 * shift + 1, (B * C, 1, 1, 2), device=device).float() * 2 / (H + 2 * shift)
        x = F.grid_sample(flat, grid + offset, padding_mode='zeros', align_corners=False).view(B, C, 3, H, W)
    mean, std = (t.to(device) for t in IMAGENET)
    return (x - mean) / std


def to_device(batch, device, shift=0):
    return (prepare_images(batch['images'], device, shift), torch.as_tensor(batch['proprio'], dtype=torch.float32, device=device),
            torch.as_tensor(batch['actions'], device=device), torch.as_tensor(batch['mask'], device=device))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', nargs='+', required=True, help='one or more demo directories'); p.add_argument('--out', required=True)
    p.add_argument('--cameras', nargs='+', default=['agentview', 'frontview', 'topview', 'left_wrist', 'right_wrist'])
    p.add_argument('--chunk', type=int, default=50); p.add_argument('--stride', type=int, default=2)
    p.add_argument('--steps', type=int, default=20000); p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--backbone-lr', type=float, default=1e-5)
    p.add_argument('--kl-weight', type=float, default=10.); p.add_argument('--shift', type=int, default=4)
    p.add_argument('--d', type=int, default=256); p.add_argument('--val-episodes', type=int, default=2)
    p.add_argument('--freeze-backbone', action='store_true', help='keep the ImageNet ResNet fixed (~2.5x faster on MPS)')
    p.add_argument('--feature-layer', type=int, default=4, choices=[3, 4], help='ResNet stage whose feature map becomes tokens')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', action='store_true', help='continue from <out>/last.pt')
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    train_eps, val_eps = split_episodes(args.data, args.val_episodes, args.seed)
    t0 = time.time()
    train = Demos(args.data, args.cameras, args.chunk, args.stride, episodes=train_eps)
    val = Demos(args.data, args.cameras, args.chunk, args.stride, episodes=val_eps)
    val.stats = train.stats
    print(f'loaded {len(train.files)} train / {len(val.files)} val episodes, {len(train)} samples in {time.time() - t0:.0f}s', flush=True)
    config = dict(cameras=train.cameras, proprio=train.proprio_keys, chunk=args.chunk, image_size=train.image_size,
                  proprio_dim=train.proprio_dim, action_dim=train.action_dim, d=args.d, feature_layer=args.feature_layer, data=args.data,
                  train_files=train.files, val_files=val.files, args=vars(args))
    model = ACT(len(train.cameras), train.proprio_dim, train.action_dim, args.chunk, train.image_size, d=args.d,
                feature_layer=args.feature_layer).to(args.device)
    if args.freeze_backbone:
        for q in model.backbone.parameters(): q.requires_grad_(False)
    backbone = [q for n, q in model.named_parameters() if n.startswith('backbone.') and q.requires_grad]
    rest = [q for n, q in model.named_parameters() if not n.startswith('backbone.')]
    groups = [dict(params=rest, lr=args.lr)] + ([dict(params=backbone, lr=args.backbone_lr)] if backbone else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    log = open(os.path.join(args.out, 'log.jsonl'), 'a')
    stats = {k: v.tolist() for k, v in train.stats.items()}

    def save(name, step):
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, config=config, stats=stats),
                   os.path.join(args.out, name))

    start = 0
    if args.resume:
        state = torch.load(os.path.join(args.out, 'last.pt'), map_location=args.device, weights_only=False)
        model.load_state_dict(state['model'])
        if 'opt' in state: opt.load_state_dict(state['opt'])
        start = state['step']
        # fresh random streams so the resumed run does not replay the first batches
        torch.manual_seed(args.seed + start); rng = np.random.default_rng(args.seed + start)
        print(f'resumed at step {start}' + ('' if 'opt' in state else ' with a fresh optimizer (none saved)'), flush=True)

    t0 = time.time()
    for step in range(start + 1, args.steps + 1):
        model.train()
        images, proprio, actions, mask = to_device(train.sample(args.batch, rng), args.device, args.shift)
        pred, mu, logvar = model(images, proprio, actions, mask)
        loss, parts = act_loss(pred, actions, mask, mu, logvar, args.kl_weight)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); opt.step()
        if step % 100 == 0 or step == 1:
            row = dict(step=step, loss=round(loss.item(), 4), **{k: round(v, 4) for k, v in parts.items()}, it_s=round((step - start) / (time.time() - t0), 2))
            if step % 1000 == 0:
                model.eval()
                with torch.no_grad():
                    vrng = np.random.default_rng(0)
                    vl = []
                    for _ in range(10):
                        images, proprio, actions, mask = to_device(val.sample(args.batch, vrng), args.device)
                        pred = model(images, proprio)[0]
                        vl.append(((pred - actions).abs().mean(-1) * mask).sum().item() / mask.sum().item())
                row['val_l1'] = round(float(np.mean(vl)), 4)
                save('last.pt', step)
            log.write(json.dumps(row) + '\n'); log.flush(); print(row, flush=True)
    save('last.pt', args.steps)
