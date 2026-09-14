#!/usr/bin/env python3
"""Confirmed Methane Plume Footprints.

Usage:  python3 solution.py <public_dir> <output_csv>

Everything below is learned from the supplied scenes alone.  A U-Net carrying
both a segmentation head and a scene-level head is trained from scratch under
K-fold cross-validation; the held-out predictions calibrate the mask threshold
and the accept/reject rule; the fold models are then ensembled over the test
split with dihedral test-time augmentation.
"""
import os

# cuBLAS needs a fixed workspace before the CUDA context exists for its
# reductions to be reproducible; this must precede the torch import.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
H = W = 672             # scene size
SEG_RES = 168           # resolution the segmentation head emits
NODATA = 65535          # reserved code: the retrieval is missing at that pixel
CY = CX = (H - 1) / 2.0

N_FOLDS = 5
EPOCHS = 40
BATCH = 8
LR = 3e-3
WD = 1e-2
W_DICE = 1.0
W_CLS = 0.6
POS_WEIGHT = 8.0
TTA = 4                 # identity plus the three in-plane flips
SEED = 1234
INFER_BATCH = 8
DEV = 'cuda'            # the execution plan is fixed, not probed

# One fixed execution plan.  No autotuner (it selects kernels by measuring
# them, which makes the arithmetic depend on machine load), no implicit
# reduced-precision matmul, and only deterministic kernels.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# run-length coding over the row-major grid, starts counted from 1
# ---------------------------------------------------------------------------
def rle_decode(s):
    m = np.zeros(H * W, np.uint8)
    if not isinstance(s, str) or s.strip() == 'no-plume':
        return m.reshape(H, W)
    v = np.array(s.split(), dtype=np.int64)
    for st, ln in zip(v[0::2], v[1::2]):
        m[st - 1:st - 1 + ln] = 1
    return m.reshape(H, W)


def rle_encode(m):
    f = np.ascontiguousarray(m).ravel().astype(np.int8)
    if not f.any():
        return 'no-plume'
    d = np.diff(np.concatenate(([0], f, [0])))
    st = np.flatnonzero(d == 1) + 1
    en = np.flatnonzero(d == -1) + 1
    return ' '.join('%d %d' % (a, b - a) for a, b in zip(st, en))


# ---------------------------------------------------------------------------
# scene channels
# ---------------------------------------------------------------------------
def scene_stats(a):
    """Robust centre and scale of the valid retrieval, plus the valid fraction."""
    v = a[a < NODATA].astype(np.float32)
    if v.size < 64:
        return 6050.0, 200.0, float(v.size) / (H * W)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    return med, max(mad, 1e-2), float(v.size) / (H * W)


_RAD = {}


def radius_ch(dev):
    if dev not in _RAD:
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing='ij')
        _RAD[dev] = (torch.hypot(yy - CY, xx - CX) / 336.0).to(dev)
    return _RAD[dev]


def build_input(u, med, mad, dev):
    """u: (B,H,W) integer codes on `dev`.  Returns a (B,5,H,W) stack.

    ch0  concentration on the one global affine map (absolute level),
    ch1  concentration standardised by the scene's own robust centre/scale,
    ch2  ch1 minus a coarse local background, i.e. the enhancement itself,
    ch3  where the retrieval succeeded,
    ch4  distance from the scene centre.
    """
    a = u.float()
    valid = (a < NODATA).float()
    med = med.view(-1, 1, 1)
    mad = mad.view(-1, 1, 1)
    zero = torch.zeros_like(a)
    z = torch.where(valid > 0, (a - med) / mad, zero).clamp(-6, 10)
    g = torch.where(valid > 0, (a - 6050.0) / 250.0, zero).clamp(-6, 6)
    zs = F.avg_pool2d((z * valid).unsqueeze(1), 8)
    vs = F.avg_pool2d(valid.unsqueeze(1), 8)
    k = 17
    zb = F.avg_pool2d(zs, k, 1, k // 2, count_include_pad=False)
    vb = F.avg_pool2d(vs, k, 1, k // 2, count_include_pad=False)
    bg = F.interpolate(zb / vb.clamp_min(1e-3), size=(H, W),
                       mode='bilinear', align_corners=False).squeeze(1)
    hp = ((z - bg) * valid).clamp(-6, 10)
    r = radius_ch(dev).expand(a.shape[0], -1, -1)
    return torch.stack([g, z, hp, valid, r], 1)


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(s, ci, co, st=1):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, st, 1, bias=False)
        s.b1 = nn.BatchNorm2d(co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        s.b2 = nn.BatchNorm2d(co)
        s.sk = None if (ci == co and st == 1) else nn.Sequential(
            nn.Conv2d(ci, co, 1, st, bias=False), nn.BatchNorm2d(co))
        s.act = nn.SiLU(inplace=True)

    def forward(s, x):
        r = x if s.sk is None else s.sk(x)
        x = s.act(s.b1(s.c1(x)))
        x = s.b2(s.c2(x))
        return s.act(x + r)


def up2(x):
    """Exact 2x nearest upsample.  Written as expand+reshape rather than
    F.interpolate because the interpolate backward accumulates with atomics,
    so it does not reproduce bit for bit.  Every decoder stage here is exactly
    2x (21 -> 42 -> 84 -> 168), so this is equivalent."""
    b, c, h, w = x.shape
    return x.view(b, c, h, 1, w, 1).expand(b, c, h, 2, w, 2).reshape(b, c, h * 2, w * 2)


class Up(nn.Module):
    def __init__(s, ci, cs, co):
        super().__init__()
        s.r = nn.Sequential(nn.Conv2d(cs, co, 1, bias=False),
                            nn.BatchNorm2d(co), nn.SiLU(inplace=True))
        s.c = nn.Sequential(
            nn.Conv2d(ci + co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.SiLU(inplace=True),
            nn.Conv2d(co, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.SiLU(inplace=True))

    def forward(s, x, sk):
        sk = s.r(sk)
        return s.c(torch.cat([up2(x), sk], 1))


class PlumeNet(nn.Module):
    """Encoder-decoder with a scene-level branch.  The two stem convolutions
    are the only full-resolution work; everything else runs at 168 px or
    below, which is what makes a five-model fold ensemble affordable."""

    def __init__(s, cin=5, w=(24, 56, 96, 160, 256), drop=0.15):
        super().__init__()
        c0, c1, c2, c3, c4 = w
        s.stem = nn.Sequential(
            nn.Conv2d(cin, c0, 5, 2, 2, bias=False), nn.BatchNorm2d(c0), nn.SiLU(inplace=True),
            nn.Conv2d(c0, c1, 3, 2, 1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(inplace=True))
        s.e1 = Block(c1, c1)
        s.e2 = nn.Sequential(Block(c1, c2, 2), Block(c2, c2))
        s.e3 = nn.Sequential(Block(c2, c3, 2), Block(c3, c3))
        s.e4 = nn.Sequential(Block(c3, c4, 2), Block(c4, c4))
        s.d3 = Up(c4, c3, c3)
        s.d2 = Up(c3, c2, c2)
        s.d1 = Up(c2, c1, 64)
        s.seg = nn.Conv2d(64, 1, 1)
        s.drop = nn.Dropout(drop)
        s.cls = nn.Sequential(nn.Linear(c4 * 2 + c3, 192), nn.SiLU(inplace=True),
                              nn.Dropout(drop), nn.Linear(192, 1))

    def forward(s, x):
        x1 = s.e1(s.stem(x))
        x2 = s.e2(x1)
        x3 = s.e3(x2)
        x4 = s.e4(x3)
        y = s.d3(x4, x3)
        y = s.d2(y, x2)
        y = s.d1(y, x1)
        f = torch.cat([x4.mean((2, 3)), x4.amax((2, 3)), x3.mean((2, 3))], 1)
        return s.seg(y), s.cls(s.drop(f)).squeeze(1)


def seg_loss(logit, tgt, pw=POS_WEIGHT):
    l = logit.squeeze(1)
    bce = F.binary_cross_entropy_with_logits(
        l, tgt, pos_weight=torch.tensor(pw, device=l.device))
    p = torch.sigmoid(l)
    has = tgt.flatten(1).sum(1) > 0
    if has.any():
        pp = p[has].flatten(1)
        tt = tgt[has].flatten(1)
        dl = (1 - (2 * (pp * tt).sum(1) + 1.0) / (pp.sum(1) + tt.sum(1) + 1.0)).mean()
    else:
        dl = l.sum() * 0
    return bce, dl


# ---------------------------------------------------------------------------
# dihedral transforms (the scene centre is a fixed point of all eight)
# ---------------------------------------------------------------------------
def d4(x, k):
    if k & 1:
        x = torch.flip(x, [-1])
    if k & 2:
        x = torch.flip(x, [-2])
    if k & 4:
        x = x.transpose(-1, -2)
    return x.contiguous()


def d4inv(x, k):
    if k & 4:
        x = x.transpose(-1, -2)
    if k & 2:
        x = torch.flip(x, [-2])
    if k & 1:
        x = torch.flip(x, [-1])
    return x


# ---------------------------------------------------------------------------
# scene-level features read off a predicted probability map
# ---------------------------------------------------------------------------
C168 = SEG_RES // 2


def label_components(m):
    """Two-pass connected components (8-connectivity) on a boolean image."""
    lab = np.zeros(m.shape, np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    hh, ww = m.shape
    for i in range(hh):
        mi = m[i]
        if not mi.any():
            continue
        for j in np.flatnonzero(mi):
            nb = []
            if j > 0 and lab[i, j - 1]:
                nb.append(lab[i, j - 1])
            if i > 0:
                for jj in (j - 1, j, j + 1):
                    if 0 <= jj < ww and lab[i - 1, jj]:
                        nb.append(lab[i - 1, jj])
            if nb:
                r = min(nb)
                lab[i, j] = r
                for q in nb:
                    union(r, q)
            else:
                lab[i, j] = nxt
                parent.append(nxt)
                nxt += 1
    if nxt == 1:
        return lab, 0
    root = np.array([find(x) for x in range(nxt)], np.int32)
    uniq, inv = np.unique(root[1:], return_inverse=True)
    remap = np.zeros(nxt, np.int32)
    remap[1:] = inv + 1
    return remap[lab], len(uniq)


FEAT_NAMES = ['pmax', 'ptop20', 'ptop100', 'ptop500', 'psum', 'pcen', 'pcen_m', 'pcen16',
              'a03', 'cen03', 'a05', 'cen05', 'a07', 'cen07',
              'ncomp', 'bigfrac', 'spread', 'fill', 'vfrac', 'zc', 'cls']


def seg_feats(p, vfrac, zc, cls):
    """p: (SEG_RES,SEG_RES) probability map."""
    v = np.sort(p.ravel())[::-1]
    f = [v[0], v[:20].mean(), v[:100].mean(), v[:500].mean(), p.sum()]
    c = p[C168 - 3:C168 + 3, C168 - 3:C168 + 3]
    f += [c.max(), c.mean(), p[C168 - 8:C168 + 8, C168 - 8:C168 + 8].mean()]
    for t in (0.3, 0.5, 0.7):
        m = p > t
        f += [np.log1p(m.sum()), float(m[C168 - 3:C168 + 3, C168 - 3:C168 + 3].any())]
    m = p > 0.5
    if m.any():
        lab, n = label_components(m)
        sz = np.bincount(lab.ravel())[1:]
        ys, xs = np.nonzero(m)
        f += [n, sz.max() / sz.sum(), np.hypot(ys - C168, xs - C168).mean(),
              m.sum() / max(1.0, (ys.max() - ys.min() + 1.0) * (xs.max() - xs.min() + 1.0))]
    else:
        f += [0.0, 0.0, 0.0, 0.0]
    f += [vfrac, zc, cls]
    return f


# ---------------------------------------------------------------------------
# a small self-contained logistic regression (batch gradient descent on a
# standardised design matrix -- deterministic, no solver-version drift)
# ---------------------------------------------------------------------------
class Logit:
    def __init__(s, l2=1.0, iters=4000, lr=0.5):
        s.l2, s.iters, s.lr = l2, iters, lr

    def fit(s, X, y):
        s.mu = X.mean(0)
        s.sd = X.std(0)
        s.sd[s.sd < 1e-9] = 1.0
        Z = (X - s.mu) / s.sd
        n, d = Z.shape
        w = np.zeros(d)
        b = 0.0
        for _ in range(s.iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            g = Z.T @ (p - y) / n + s.l2 * w / n
            gb = (p - y).mean()
            w -= s.lr * g
            b -= s.lr * gb
        s.w, s.b = w, b
        return s

    def predict_proba(s, X):
        Z = (X - s.mu) / s.sd
        return 1.0 / (1.0 + np.exp(-(Z @ s.w + s.b)))


def kfold_indices(y, k, seed):
    """Stratified folds, deterministic given (y, k, seed)."""
    rng = np.random.RandomState(seed)
    fold = np.zeros(len(y), np.int64)
    for c in (0, 1):
        i = np.flatnonzero(y == c)
        i = i[rng.permutation(len(i))]
        fold[i] = np.arange(len(i)) % k
    return fold


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'submission.csv'
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    log('device', DEV)

    tr = pd.read_csv(os.path.join(root, 'train.csv'))
    te = pd.read_csv(os.path.join(root, 'test.csv'))
    ntr, nte = len(tr), len(te)
    log('train %d  test %d' % (ntr, nte))

    # ---- read every scene once -------------------------------------------
    rel = list(tr['scene']) + list(te['scene'])
    imgs = np.zeros((ntr + nte, H, W), np.uint16)

    def _load(i):
        imgs[i] = np.asarray(Image.open(os.path.join(root, rel[i])), dtype=np.uint16)

    for i in range(ntr + nte):
        _load(i)
    log('scenes read', ntr + nte)

    stats = np.array([scene_stats(imgs[i]) for i in range(ntr + nte)], np.float32)
    stats_t = torch.from_numpy(stats)

    def batch_input(b):
        u = torch.from_numpy(imgs[b].astype(np.int32)).to(DEV)
        return build_input(u, stats_t[b, 0].to(DEV), stats_t[b, 1].to(DEV), DEV)

    # central enhancement level, used as one scene-level feature
    zc = np.zeros(ntr + nte, np.float32)
    for i in range(ntr + nte):
        a = imgs[i, 300:372, 300:372].astype(np.float32)
        ok = a < NODATA
        zc[i] = ((a[ok] - stats[i, 0]) / stats[i, 1]).mean() if ok.any() else 0.0

    # ---- targets ----------------------------------------------------------
    y = (tr['mask_rle'].astype(str) != 'no-plume').values.astype(np.float32)
    masks = np.zeros((ntr, H, W), np.uint8)
    for i, s in enumerate(tr['mask_rle'].values):
        masks[i] = rle_decode(s)
    soft = torch.zeros(ntr, SEG_RES, SEG_RES)
    for i in range(0, ntr, 64):
        b = torch.from_numpy(masks[i:i + 64]).float().unsqueeze(1)
        soft[i:i + 64] = F.avg_pool2d(b, H // SEG_RES).squeeze(1)
    yt = torch.from_numpy(y)
    log('targets built  positives %.3f' % y.mean())

    # ---- cross-validated training ----------------------------------------
    fold = kfold_indices(y.astype(np.int64), N_FOLDS, SEED)
    oof_p = np.zeros((ntr, SEG_RES, SEG_RES), np.float16)
    oof_c = np.zeros(ntr, np.float32)
    test_p = np.zeros((nte, SEG_RES, SEG_RES), np.float32)
    test_c = np.zeros(nte, np.float32)
    te_idx = np.arange(ntr, ntr + nte)

    @torch.no_grad()
    def predict(model, idx):
        model.eval()
        P = np.zeros((len(idx), SEG_RES, SEG_RES), np.float32)
        C = np.zeros(len(idx), np.float32)
        for i in range(0, len(idx), INFER_BATCH):
            b = idx[i:i + INFER_BATCH]
            x = batch_input(b)
            acc = 0
            cacc = 0
            for k in range(TTA):
                with torch.autocast('cuda', torch.float16):
                    sg, cl = model(d4(x, k))
                acc = acc + torch.sigmoid(d4inv(sg.float(), k))
                cacc = cacc + torch.sigmoid(cl.float())
            P[i:i + len(b)] = (acc / TTA).squeeze(1).cpu().numpy()
            C[i:i + len(b)] = (cacc / TTA).cpu().numpy()
        return P, C

    for f in range(N_FOLDS):
        tri = np.flatnonzero(fold != f)
        vai = np.flatnonzero(fold == f)
        torch.manual_seed(SEED + 17 * f)
        np.random.seed(SEED + 17 * f)
        model = PlumeNet().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        nb = len(tri) // BATCH
        steps = EPOCHS * nb
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.25)
        scaler = torch.amp.GradScaler()
        step = 0
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(tri)
            for it in range(nb):
                b = perm[it * BATCH:(it + 1) * BATCH]
                x = batch_input(b)
                m = soft[b].to(DEV)
                k = np.random.randint(8)
                x = d4(x, k)
                m = d4(m, k)
                # contrast jitter on the enhancement channels, and a level
                # jitter on ch0 so no absolute concentration offset can be
                # memorised as a shortcut
                x[:, 1:3] *= torch.empty(x.shape[0], 1, 1, 1, device=DEV).uniform_(0.9, 1.1)
                x[:, 0:1] += torch.empty(x.shape[0], 1, 1, 1, device=DEV).uniform_(-0.5, 0.5)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', torch.float16):
                    sg, cl = model(x)
                    bce, dl = seg_loss(sg, m)
                    loss = bce + W_DICE * dl + W_CLS * F.binary_cross_entropy_with_logits(cl, yt[b].to(DEV))
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                if step < steps - 1:
                    sch.step()
                step += 1
        P, C = predict(model, vai)
        oof_p[vai] = P.astype(np.float16)
        oof_c[vai] = C
        P, C = predict(model, te_idx)
        test_p += P / N_FOLDS
        test_c += C / N_FOLDS
        log('fold %d of %d done' % (f + 1, N_FOLDS))
        del model, opt
        torch.cuda.empty_cache()

    # ---- calibrate the mask threshold on held-out positives ---------------
    def up672(p):
        t = torch.from_numpy(np.ascontiguousarray(p).astype(np.float32)).unsqueeze(1)
        return F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False).squeeze(1).numpy()

    pos = np.flatnonzero(y > 0)
    best_t, best_d = 0.6, -1.0
    grid = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    sums = np.zeros(len(grid))
    for i0 in range(0, len(pos), 64):
        chunk = pos[i0:i0 + 64]
        U = up672(oof_p[chunk])
        for gi, t in enumerate(grid):
            for j, q in enumerate(chunk):
                pm = U[j] > t
                tt = masks[q].astype(bool)
                sums[gi] += 2.0 * (pm & tt).sum() / max(1, pm.sum() + tt.sum())
    for gi, t in enumerate(grid):
        d = sums[gi] / len(pos)
        log('  mask threshold %.2f -> held-out Dice %.4f' % (t, d))
        if d > best_d:
            best_d, best_t = d, t
    log('mask threshold %.2f  Dice %.4f' % (best_t, best_d))

    # per-scene Dice at the chosen threshold, needed to score decision rules
    dvec = np.zeros(ntr)
    for i0 in range(0, len(pos), 64):
        chunk = pos[i0:i0 + 64]
        U = up672(oof_p[chunk])
        for j, q in enumerate(chunk):
            pm = U[j] > best_t
            tt = masks[q].astype(bool)
            dvec[q] = 2.0 * (pm & tt).sum() / max(1, pm.sum() + tt.sum())

    # ---- scene-level decision --------------------------------------------
    Xtr = np.array([seg_feats(oof_p[i].astype(np.float32), stats[i, 2], zc[i], oof_c[i])
                    for i in range(ntr)], np.float64)
    Xte = np.array([seg_feats(test_p[i], stats[ntr + i, 2], zc[ntr + i], test_c[i])
                    for i in range(nte)], np.float64)

    # rank positions are read off the training distribution so that train and
    # test features land on one common scale
    ref = np.sort(Xtr, axis=0)
    def expand(M):
        r = np.empty_like(M)
        for j in range(M.shape[1]):
            r[:, j] = np.searchsorted(ref[:, j], M[:, j], side='left') / float(len(ref))
        return np.column_stack([M, r, r ** 2])

    Etr, Ete = expand(Xtr), expand(Xte)
    inner = kfold_indices(y.astype(np.int64), 5, SEED + 99)
    score_tr = np.zeros(ntr)
    for f in range(5):
        a = np.flatnonzero(inner != f)
        b = np.flatnonzero(inner == f)
        score_tr[b] = Logit().fit(Etr[a], y[a]).predict_proba(Etr[b])
    score_te = Logit().fit(Etr, y).predict_proba(Ete)

    # pick the fraction of scenes to accept that maximises the held-out score
    negfrac = 1.0 - y.mean()
    best_q, best_s = 1.0 - negfrac, -1.0
    for q in np.arange(0.35, 0.90, 0.005):
        thr = np.quantile(score_tr, 1.0 - q)
        pred = (score_tr > thr).astype(float)
        dec = (pred == y).astype(float)
        R = 0.5 * dec.mean() + 0.5 * (dvec[pos] * pred[pos]).mean()
        s = (R - 0.5 * negfrac) / (1.0 - 0.5 * negfrac)
        if s > best_s:
            best_s, best_q = s, q
    log('accept fraction %.3f  held-out challenge score %.4f' % (best_q, best_s))

    keep = score_te > np.quantile(score_te, 1.0 - best_q)

    # ---- write the submission --------------------------------------------
    rows = []
    for i0 in range(0, nte, 64):
        chunk = np.arange(i0, min(i0 + 64, nte))
        U = up672(test_p[chunk])
        for j, q in enumerate(chunk):
            if not keep[q]:
                rows.append('no-plume')
                continue
            m = (U[j] > best_t) & (imgs[ntr + q] < NODATA)
            if not m.any():
                # accepted scene: fall back to the most probable pixels at the
                # centre, since every confirmed footprint covers the centre
                v = np.where(imgs[ntr + q] < NODATA, U[j], -1.0)
                a, b = np.unravel_index(int(np.argmax(v)), (H, W))
                m = np.zeros((H, W), bool)
                m[max(0, a - 4):a + 5, max(0, b - 4):b + 5] = True
                m &= imgs[ntr + q] < NODATA
                if not m.any():
                    m[a, b] = True
            rows.append(rle_encode(m))

    sub = pd.DataFrame({'sample_id': te['sample_id'].values, 'mask_rle': rows})
    sub.to_csv(out_path, index=False)
    log('wrote %s  rows %d  non-empty %d'
        % (out_path, len(sub), int((sub['mask_rle'] != 'no-plume').sum())))


if __name__ == '__main__':
    main()
