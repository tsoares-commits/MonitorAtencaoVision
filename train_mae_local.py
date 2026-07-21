"""
YOLO -> FPN -> Tokens -> MAE (auto-supervisionado) — versao local Windows/Linux.

Modos (rode em ordem):
    python train_mae_local.py extract   # 1x: extrai features YOLO P3/P4/P5 para disco
    python train_mae_local.py train     # treina MAE usando cache
    python train_mae_local.py both      # extract + train

Ajuste o bloco CONFIG abaixo. Coloque o .pt do YOLO e a pasta de frames
no mesmo diretorio do script (ou edite os caminhos).
"""

import glob
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm


# =========================
# CONFIG
# =========================
HERE = Path(__file__).resolve().parent

YOLO_WEIGHTS = HERE / "best.pt"                              # << seu .pt do YOLO
FRAMES_DIR   = HERE / "frames_pool_filtered"                 # << frames inteiros, filtrados pela deteccao YOLO
CACHE_DIR    = HERE / "cache_yolo_feats_sobel_box"           # << cache p/ Sobel*Box (mesmo da run anterior)
SAVE_DIR     = HERE / "ckpt_mae_attn"                        # << NOVO: ckpts p/ run com attention supervision

IMG_SIZE     = 640                         # mantem P3=80x80 (stride 8 do YOLOv8)
BATCH_SIZE   = 8                           # CPU + 32GB RAM aguenta 8 com EMBED_DIM=768; reduza se SWAP
EPOCHS       = 20
LR           = 2e-4
WEIGHT_DECAY = 0.05
MASK_RATIO   = 0.75                        # 25% visiveis = 6 patches (top-K por gradiente Sobel)
NUM_WORKERS  = 2                           # workers prefetcham .pt enquanto CPU calcula
SAVE_EVERY   = 5                           # salva checkpoint a cada N epocas (e o ultimo sempre)

# === Mascaramento determinístico (Sobel + YOLO box) ===
USE_SOBEL_MASK = True
USE_BOX_MASK   = True
BOTTOM_ROWS    = 5

# === Attention Supervision (Etapa 2 - dual forward pass) ===
# Adiciona um termo na loss que PENALIZA a atencao do encoder em patches
# fora da box do YOLO e RECOMPENSA dentro. Funciona via 2 forwards do encoder
# por imagem (pass 1 = MAE com mask, pass 2 = encoder em todos 25 tokens).
ATTN_LAMBDA    = 0.5     # peso do termo de atencao na loss total (0 = desliga, 1 = igual a recon)
INSIDE_WEIGHT  = 1.0     # peso da diferenca para patches DENTRO da box
OUTSIDE_WEIGHT = 2.0     # peso da diferenca para patches FORA da box (penalizacao)

# indices das camadas P3/P4/P5 (YOLOv8m).
# Confira em: list(yolo.model.model)
IDX_P3, IDX_P4, IDX_P5 = 15, 18, 21

# MAE
EMBED_DIM    = 384                         # cortar para 384 reduz custo da FFN ~4x
NUM_HEADS    = 4                           # talvez aumentar para 6 
ENC_DEPTH    = 2
DEC_DEPTH    = 1
MLP_RATIO    = 4.0
DROPOUT      = 0.1
PATCH_SIZE   = 16

# CPU tuning
# Core 7 250U nao tem AVX-512_BF16/AMX -> autocast(bf16) so cria overhead
USE_BF16_CPU = False
# CPU hibrida (P-cores + E-cores): usar SO os P-cores costuma ser mais rapido
# que misturar com E-cores. Core 7 250U tem 2 P-cores * 2 threads HT = 4.
NUM_THREADS  = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_runtime():
    if DEVICE.type == "cpu":
        torch.set_num_threads(NUM_THREADS)
        try:
            torch.set_num_interop_threads(max(1, NUM_THREADS // 2))
        except RuntimeError:
            pass
        try:
            torch.backends.mkldnn.enabled = True
        except Exception:
            pass


# =========================
# Util: Sobel patch scores (usado p/ mascaramento deterministico no MAE)
# =========================
def compute_sobel_patch_scores(img_rgb_uint8, grid=5, bottom_rows=5):
    """Score Sobel agregado por patch num grid GxG, restrito as bottom_rows
    fileiras inferiores (top rows ficam em zero).

    img_rgb_uint8: HxWx3 uint8 RGB (apos resize para imgsz)
    grid: 5 -> 25 patches
    bottom_rows: 5 -> usa todas as fileiras (default novo)
                 2 -> so as 2 fileiras de baixo (comportamento antigo)

    Retorna tensor [grid*grid] = [25] em float32 com scores; entradas das
    fileiras superiores (acima de bottom_rows) ficam 0.
    """
    gray = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = gray.shape
    h_step = H // grid
    w_step = W // grid

    mag = np.zeros((h_step * grid, w_step * grid), dtype=np.float32)
    y_start = (grid - bottom_rows) * h_step
    y_end = grid * h_step
    strip = gray[y_start:y_end, :w_step * grid]
    gx = cv2.Sobel(strip, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(strip, cv2.CV_32F, 0, 1, ksize=3)
    mag[y_start:y_end, :] = gx * gx + gy * gy

    patches = mag.reshape(grid, h_step, grid, w_step).mean(axis=(1, 3))
    return torch.from_numpy(patches.flatten().astype(np.float32))   # [25]


def box_to_patch_mask_5x5(box_xyxy, img_size=640, grid=5):
    """Mascara binaria [grid*grid] = [25] indicando quais patches do grid
    INTERSECTAM a bounding box do YOLO (1) ou nao (0).

    box_xyxy: tupla (x1, y1, x2, y2) em pixels da imagem (640 default)
    img_size: 640
    grid: 5
    """
    x1, y1, x2, y2 = box_xyxy
    h_step = img_size // grid
    w_step = img_size // grid
    mask = np.zeros(grid * grid, dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            patch_x1 = j * w_step
            patch_x2 = (j + 1) * w_step
            patch_y1 = i * h_step
            patch_y2 = (i + 1) * h_step
            # interseccao retangular: x1 < patch_x2 AND x2 > patch_x1 (idem y)
            if (x1 < patch_x2 and x2 > patch_x1 and
                    y1 < patch_y2 and y2 > patch_y1):
                mask[i * grid + j] = 1.0
    return mask


# =========================
# Datasets
# =========================
class FramesUnlabeled(Dataset):
    EXTS = (
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF",
    )

    def __init__(self, frames_dir, imgsz=640, recursive=True):
        self.imgsz = imgsz
        root = Path(frames_dir)
        if not root.exists():
            raise FileNotFoundError(f"Pasta nao existe: {root}")
        files = set()
        for e in self.EXTS:
            if recursive:
                files.update(glob.glob(str(root / "**" / e), recursive=True))
            else:
                files.update(glob.glob(str(root / e)))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada em: {root} (busca recursiva). "
                f"Confira se as imagens estao la e se a extensao bate."
            )
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return x, Path(path).stem


class CachedFeatures(Dataset):
    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(str(Path(cache_dir) / "*.pt")))
        if not self.files:
            raise FileNotFoundError(f"Cache vazio em {cache_dir}. Rode 'extract' primeiro.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            d = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        except TypeError:
            d = torch.load(self.files[idx], map_location="cpu")
        # patch_scores pode nao existir em caches antigos -> retorna zeros nesse caso
        scores = d.get("patch_scores", torch.zeros(25, dtype=torch.float32))
        return d["P3"].float(), d["P4"].float(), d["P5"].float(), scores.float()


# =========================
# Modelos
# =========================
class ExternalFPN(nn.Module):
    def __init__(self, c3, c4, c5, outc=64):
        super().__init__()
        self.p3 = nn.Conv2d(c3, outc, 1)
        self.p4 = nn.Conv2d(c4, outc, 1)
        self.p5 = nn.Conv2d(c5, outc, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, P3, P4, P5):
        P3 = self.act(self.p3(P3))
        P4 = self.act(self.p4(P4))
        P5 = self.act(self.p5(P5))
        P4 = F.interpolate(P4, size=P3.shape[-2:], mode="bilinear", align_corners=False)
        P5 = F.interpolate(P5, size=P3.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([P3, P4, P5], dim=1)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim=768, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x2 = self.ln1(x)
        a, _ = self.attn(x2, x2, x2, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class TokenMAE(nn.Module):
    def __init__(self, in_c3, in_c4, in_c5, fpn_outc=64, embed_dim=768, patch_size=16,
                 enc_depth=2, dec_depth=1, heads=6, mlp_ratio=4.0, drop=0.1, fmap_size=80):
        super().__init__()
        self.fpn = ExternalFPN(in_c3, in_c4, in_c5, outc=fpn_outc)
        self.patch = PatchEmbedding(in_channels=fpn_outc * 3, embed_dim=embed_dim, patch_size=patch_size)
        self.fmap_size = fmap_size

        self.num_patches = (fmap_size // patch_size) ** 2
        self.pos_enc = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)

        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(enc_depth)
        ])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos_dec = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_dec, std=0.02)

        self.decoder = nn.Sequential(*[
            TransformerBlock(embed_dim, heads, mlp_ratio=mlp_ratio, drop=drop) for _ in range(dec_depth)
        ])

        self.pred = nn.Linear(embed_dim, embed_dim)

    def _encoder_with_attn(self, x):
        """Roda o encoder bloco a bloco, capturando a atencao do ULTIMO bloco.
        Retorna (saida, attn_last) onde attn_last e [B, H, N, N]."""
        last_idx = len(self.encoder) - 1
        attn_last = None
        for i, blk in enumerate(self.encoder):
            x2 = blk.ln1(x)
            a, w = blk.attn(x2, x2, x2, need_weights=True, average_attn_weights=False)
            if i == last_idx:
                attn_last = w  # [B, H, N, N]
            x = x + a
            x = x + blk.ffn(blk.ln2(x))
        return x, attn_last

    def forward(self, P3, P4, P5, mask_ratio=0.75, patch_scores=None,
                box_mask=None, attn_lambda=0.0,
                inside_weight=1.0, outside_weight=2.0):
        """
        patch_scores : [B, N] - se passado, top-K determina visiveis (Sobel*Box).
        box_mask     : [B, N] - 1 dentro da box do YOLO, 0 fora. Se passado E
                       attn_lambda > 0, ativa o pass 2 (attention supervision).
        attn_lambda  : peso do termo de supervisao de atencao.
        inside_weight: peso da diferenca para patches DENTRO da box.
        outside_weight: peso da diferenca para patches FORA da box (penalizacao).

        Retorna dict {'loss', 'loss_recon', 'loss_attn'} para logs detalhados.
        """
        F_out = self.fpn(P3, P4, P5)
        F_out = F.interpolate(F_out, size=(self.fmap_size, self.fmap_size),
                              mode="bilinear", align_corners=False)

        F_out = F_out.float()
        F_out = (F_out - F_out.mean(dim=[2, 3], keepdim=True)) / (F_out.std(dim=[2, 3], keepdim=True) + 1e-6)

        tokens = self.patch(F_out)  # [B, N, D]
        B, N, D = tokens.shape

        # =================== ETAPA 1: MAE classico com mask ===================
        len_keep = int(N * (1 - mask_ratio))
        if patch_scores is not None:
            ids_shuffle = torch.argsort(patch_scores, dim=1, descending=True)
        else:
            noise = torch.rand(B, N, device=tokens.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_mask = ids_shuffle[:, len_keep:]

        x_vis = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        pos_vis = torch.gather(self.pos_enc.expand(B, -1, -1), dim=1,
                               index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        x_vis = x_vis + pos_vis

        z = self.encoder(x_vis)

        mask_tokens = self.mask_token.expand(B, N - len_keep, D)
        z_ = torch.cat([z, mask_tokens], dim=1)
        z_full = torch.gather(z_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        z_full = z_full + self.pos_dec

        dec = self.decoder(z_full)
        pred = self.pred(dec)

        target = tokens.detach()
        t_mean = target.mean(dim=-1, keepdim=True)
        t_std = target.std(dim=-1, keepdim=True)
        target = (target - t_mean) / (t_std + 1e-6)

        pred_mask = torch.gather(pred, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        tgt_mask = torch.gather(target, dim=1, index=ids_mask.unsqueeze(-1).expand(-1, -1, D))
        loss_recon = F.mse_loss(pred_mask, tgt_mask)

        # =================== ETAPA 2: Attention Supervision ===================
        if box_mask is not None and attn_lambda > 0:
            # encoder ve TODOS os 25 tokens (sem mask), captura atencao do ultimo bloco
            x_full = tokens + self.pos_enc.expand(B, -1, -1)
            _, attn_last = self._encoder_with_attn(x_full)   # [B, H, N, N]

            # atencao recebida por patch = media sobre heads e queries
            attn_recv = attn_last.mean(dim=1).mean(dim=1)    # [B, N]
            # normaliza como distribuicao de probabilidade (soma = 1)
            attn_dist = attn_recv / (attn_recv.sum(dim=1, keepdim=True) + 1e-9)

            # alvo: distribuicao uniforme dentro da box, 0 fora
            bm = box_mask.float()
            target_dist = bm / (bm.sum(dim=1, keepdim=True) + 1e-9)

            # weighted MSE: penalidade DIFERENTE dentro vs fora
            weights = inside_weight * bm + outside_weight * (1.0 - bm)
            loss_attn = (weights * (attn_dist - target_dist).pow(2)).mean()

            loss_total = loss_recon + attn_lambda * loss_attn
        else:
            loss_attn = torch.zeros((), device=tokens.device)
            loss_total = loss_recon

        return {"loss": loss_total, "loss_recon": loss_recon, "loss_attn": loss_attn}


# =========================
# Phase 1: extract features
# =========================
def extract_features():
    from ultralytics import YOLO
    setup_runtime()
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    print(f"[extract] Device: {DEVICE} | YOLO: {YOLO_WEIGHTS}")
    if not Path(YOLO_WEIGHTS).exists():
        raise FileNotFoundError(f"YOLO weights nao encontrado: {YOLO_WEIGHTS}")

    yolo = YOLO(str(YOLO_WEIGHTS))
    net = yolo.model
    net.eval().to(DEVICE)
    for p in net.parameters():
        p.requires_grad = False

    feats = {}
    def hook(name):
        def _hook(_m, _inp, out):
            feats[name] = out
        return _hook

    h3 = net.model[IDX_P3].register_forward_hook(hook("P3"))
    h4 = net.model[IDX_P4].register_forward_hook(hook("P4"))
    h5 = net.model[IDX_P5].register_forward_hook(hook("P5"))

    ds = FramesUnlabeled(FRAMES_DIR, imgsz=IMG_SIZE)
    print(f"[extract] {len(ds)} frames")

    skipped_existing = 0
    skipped_no_det = 0
    saved = 0
    for i in tqdm(range(len(ds)), desc="extract"):
        x, name = ds[i]
        out_path = Path(CACHE_DIR) / f"{name}.pt"
        if out_path.exists():
            skipped_existing += 1
            continue

        # converte tensor [0,1] RGB para uint8 RGB e BGR
        img_rgb_uint8 = (x.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)

        # roda YOLO via predict (dispara hooks p/ features + retorna boxes parseadas)
        feats.clear()
        try:
            results = yolo.predict(
                source=img_bgr, imgsz=IMG_SIZE, conf=0.25,
                verbose=False, device=str(DEVICE),
            )
        except Exception:
            skipped_no_det += 1
            continue

        # exige que o YOLO tenha detectado algo (deveria, frames_pool_filtered ja filtrou)
        if not results or len(results[0].boxes) == 0:
            skipped_no_det += 1
            continue
        rb = results[0].boxes
        xyxy = rb.xyxy.cpu().numpy()
        conf_arr = rb.conf.cpu().numpy()
        idx_max = int(np.argmax(conf_arr))
        bx1, by1, bx2, by2 = xyxy[idx_max].astype(int)

        # confere que os hooks pegaram as features
        if "P3" not in feats or "P4" not in feats or "P5" not in feats:
            skipped_no_det += 1
            continue

        # Sobel scores (todas as fileiras se BOTTOM_ROWS=5)
        sobel_scores = compute_sobel_patch_scores(
            img_rgb_uint8, grid=5, bottom_rows=BOTTOM_ROWS
        ).numpy()

        # box mask: 1 nos patches dentro da box do YOLO
        if USE_BOX_MASK:
            box_mask = box_to_patch_mask_5x5(
                (bx1, by1, bx2, by2), img_size=IMG_SIZE, grid=5
            )
            constrained = sobel_scores * box_mask
            # fallback: se a interseccao box ∩ bottom_rows < 6 patches, usa box_mask puro
            # (todos patches da box ficam empatados, top-K pega 6 deles)
            if (constrained > 0).sum() < 6:
                constrained = box_mask
        else:
            constrained = sobel_scores

        d = {
            "P3": feats["P3"].squeeze(0).detach().to("cpu", torch.float16).contiguous(),
            "P4": feats["P4"].squeeze(0).detach().to("cpu", torch.float16).contiguous(),
            "P5": feats["P5"].squeeze(0).detach().to("cpu", torch.float16).contiguous(),
            "patch_scores": torch.from_numpy(constrained.astype(np.float32)).contiguous(),
        }
        torch.save(d, out_path)
        saved += 1

    h3.remove(); h4.remove(); h5.remove()
    print(f"[extract] OK. Salvos: {saved} | pulados (ja existiam): {skipped_existing} | "
          f"sem deteccao YOLO: {skipped_no_det}")
    print(f"           patch_scores: Sobel({BOTTOM_ROWS} bottom rows) "
          f"{'* box_mask_YOLO' if USE_BOX_MASK else '(sem constraint de box)'}")
    print(f"           cache: {CACHE_DIR}")


# =========================
# Phase 2: train MAE
# =========================
def train_mae():
    setup_runtime()
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    ds = CachedFeatures(CACHE_DIR)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=(2 if NUM_WORKERS > 0 else None),
    )
    print(f"[train] frames em cache: {len(ds)} | batches/epoch: {len(dl)}")

    P3_0, P4_0, P5_0, scores_0 = ds[0]
    c3, c4, c5 = P3_0.shape[0], P4_0.shape[0], P5_0.shape[0]
    fmap_size = P3_0.shape[-1]
    print(f"[train] canais detectados: c3={c3} c4={c4} c5={c5} | fmap_size={fmap_size}")
    has_scores = bool((scores_0.abs().sum() > 0))
    print(f"[train] patch_scores no cache: {'SIM' if has_scores else 'NAO (cache antigo)'}  "
          f"| USE_SOBEL_MASK={USE_SOBEL_MASK}")

    mae = TokenMAE(
        in_c3=c3, in_c4=c4, in_c5=c5,
        fpn_outc=64, embed_dim=EMBED_DIM, patch_size=PATCH_SIZE,
        enc_depth=ENC_DEPTH, dec_depth=DEC_DEPTH,
        heads=NUM_HEADS, mlp_ratio=MLP_RATIO, drop=DROPOUT,
        fmap_size=fmap_size,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(mae.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    use_cuda = (DEVICE.type == "cuda")
    scaler = GradScaler(enabled=use_cuda)

    print("\n===== CONFIG TREINO =====")
    print(f"DEVICE={DEVICE} | EPOCHS={EPOCHS} | BS={BATCH_SIZE} | LR={LR} | WD={WEIGHT_DECAY} | MASK={MASK_RATIO}")
    print(f"EMBED_DIM={EMBED_DIM} | HEADS={NUM_HEADS} | ENC_DEPTH={ENC_DEPTH} | DEC_DEPTH={DEC_DEPTH}")
    print(f"BF16_CPU={USE_BF16_CPU} | THREADS={NUM_THREADS} | NUM_WORKERS={NUM_WORKERS}")
    if USE_SOBEL_MASK and has_scores:
        print(f"MASK MODE = Sobel-deterministico (top-6 nas {BOTTOM_ROWS} fileiras inferiores)")
    else:
        print("MASK MODE = aleatorio (MAE classico)")
    if ATTN_LAMBDA > 0 and has_scores:
        print(f"ATTN SUPERV = ON | lambda={ATTN_LAMBDA} | inside_w={INSIDE_WEIGHT} outside_w={OUTSIDE_WEIGHT}")
    else:
        print("ATTN SUPERV = OFF")
    print("=========================\n")

    log_path = Path(SAVE_DIR) / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,epoch,avg_loss,avg_recon,avg_attn,batches,time_s,cum_time_s\n")
    run_start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"# run_start={run_start_ts} EMBED_DIM={EMBED_DIM} HEADS={NUM_HEADS} "
                f"BS={BATCH_SIZE} LR={LR} EPOCHS={EPOCHS} "
                f"ATTN_LAMBDA={ATTN_LAMBDA} INSIDE_W={INSIDE_WEIGHT} OUTSIDE_W={OUTSIDE_WEIGHT}\n")
    cum_time = 0.0

    mae.train()
    for epoch in range(EPOCHS):
        t0 = time.time()
        running = 0.0
        n = 0
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        running_recon = 0.0
        running_attn = 0.0
        for P3, P4, P5, scores in pbar:
            P3 = P3.to(DEVICE, non_blocking=True)
            P4 = P4.to(DEVICE, non_blocking=True)
            P5 = P5.to(DEVICE, non_blocking=True)
            # passa patch_scores apenas se a flag estiver ativa E o cache traz scores
            scores_dev = scores.to(DEVICE, non_blocking=True) if has_scores else None
            ps = scores_dev if (USE_SOBEL_MASK and has_scores) else None
            # box_mask derivado dos patch_scores: posicoes > 0 = dentro da box
            bm = (scores_dev > 0).float() if (ATTN_LAMBDA > 0 and has_scores) else None

            optimizer.zero_grad(set_to_none=True)

            mae_kwargs = dict(
                mask_ratio=MASK_RATIO, patch_scores=ps,
                box_mask=bm, attn_lambda=ATTN_LAMBDA,
                inside_weight=INSIDE_WEIGHT, outside_weight=OUTSIDE_WEIGHT,
            )

            if use_cuda:
                with autocast(device_type="cuda", enabled=True):
                    out = mae(P3, P4, P5, **mae_kwargs)
                scaler.scale(out["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast(device_type="cpu", dtype=torch.bfloat16, enabled=USE_BF16_CPU):
                    out = mae(P3, P4, P5, **mae_kwargs)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(mae.parameters(), max_norm=1.0)
                optimizer.step()

            running += out["loss"].item()
            running_recon += out["loss_recon"].item()
            running_attn += out["loss_attn"].item()
            n += 1
            pbar.set_postfix({
                "tot":   f"{out['loss'].item():.3f}",
                "recon": f"{out['loss_recon'].item():.3f}",
                "attn":  f"{out['loss_attn'].item():.3f}",
            })

        avg = running / max(n, 1)
        avg_recon = running_recon / max(n, 1)
        avg_attn = running_attn / max(n, 1)
        dt = time.time() - t0
        cum_time += dt
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | tot={avg:.4f} recon={avg_recon:.4f} attn={avg_attn:.4f} "
              f"| batches={n} | time={dt:.1f}s | cum={cum_time:.1f}s")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ts},{epoch+1},{avg:.6f},{avg_recon:.6f},{avg_attn:.6f},{n},{dt:.1f},{cum_time:.1f}\n")

        is_last = (epoch + 1) == EPOCHS
        if is_last or ((epoch + 1) % SAVE_EVERY == 0):
            ckpt_path = Path(SAVE_DIR) / f"mae_epoch_{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch + 1,
                "avg_loss": avg,
                "state_dict": mae.state_dict(),
                "opt": optimizer.state_dict(),
                "config": {
                    "EMBED_DIM": EMBED_DIM, "NUM_HEADS": NUM_HEADS,
                    "ENC_DEPTH": ENC_DEPTH, "DEC_DEPTH": DEC_DEPTH,
                    "MLP_RATIO": MLP_RATIO, "DROPOUT": DROPOUT,
                    "MASK_RATIO": MASK_RATIO, "BATCH_SIZE": BATCH_SIZE,
                    "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
                    "PATCH_SIZE": PATCH_SIZE, "IMG_SIZE": IMG_SIZE,
                    "c3": c3, "c4": c4, "c5": c5, "fmap_size": fmap_size,
                },
            }, ckpt_path)

    print("Treino MAE concluido. Checkpoints em:", SAVE_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["extract", "train", "both"])
    args = parser.parse_args()

    if args.mode in ("extract", "both"):
        extract_features()
    if args.mode in ("train", "both"):
        train_mae()


if __name__ == "__main__":
    main()
