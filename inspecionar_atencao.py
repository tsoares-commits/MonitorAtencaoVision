r"""
inspecionar_atencao.py
----------------------
Gera heatmaps de atencao do encoder MAE treinado sobreposto a imagem original.

Para cada head do ULTIMO bloco do encoder, calcula a "atencao recebida" por cada
patch (media sobre as queries) e desenha como heatmap em cima da imagem 640x640.

Modos de uso:

    # Pasta inteira (default: Imagens_Test\ ao lado do script)
    python inspecionar_atencao.py
    python inspecionar_atencao.py --dir caminho\outra_pasta

    # Imagem unica
    python inspecionar_atencao.py --img caminho\foto.jpg

    # Ajustes
    python inspecionar_atencao.py --alpha 0.6
    python inspecionar_atencao.py --mae ckpt_mae\mae_epoch_005.pt

Saidas (uma subpasta por imagem dentro de inspecao_lote\):
    inspecao_lote\<nome_da_imagem>\
        00_original.png      - imagem 640x640 sem overlay
        head_N.png           - heatmap da head N
        mean_heads.png       - heatmap medio entre as heads
        panel.png            - tudo lado a lado
"""

import argparse
import importlib.util
from importlib.machinery import SourceFileLoader
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch  # noqa: F401  (carregado indiretamente pelo encoder)
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "Imagens_Test"
DEFAULT_MAE_PT_CROPS = HERE / "ckpt_mae_crops" / "mae_epoch_020.pt"  # usado quando --crop_with_yolo

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF")


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _import_encoder_module():
    """Importa EncoderMAE mesmo se o arquivo nao tiver .py / tiver acento."""
    candidates = [
        HERE / "aplicacao_encoder.py",
        HERE / "Aplicacao_Enconder",
        HERE / "Aplicacao_Encoder",
        HERE / "Aplicacao_Enconder.py",
        HERE / "Aplicação_Enconder",
    ]
    for c in candidates:
        if c.exists():
            # forca SourceFileLoader pra tratar como Python mesmo sem extensao .py
            loader = SourceFileLoader("aplic_enc", str(c))
            spec = importlib.util.spec_from_loader("aplic_enc", loader)
            if spec is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
            return mod.EncoderMAE
    raise FileNotFoundError(
        "Nao encontrei o arquivo do encoder (Aplicacao_Enconder ou aplicacao_encoder.py)."
    )


def _heatmap_overlay(img_rgb, heat_small, alpha=0.5, cmap=cv2.COLORMAP_JET):
    """Sobrepoe heatmap (matriz pequena, ex 5x5) na imagem.

    img_rgb: HxWx3 uint8 RGB
    heat_small: array 2D float (geralmente 5x5)
    """
    h = heat_small.astype(np.float32)
    rng = h.max() - h.min()
    h = (h - h.min()) / (rng + 1e-9)
    H, W = img_rgb.shape[:2]
    h_full = cv2.resize(h, (W, H), interpolation=cv2.INTER_CUBIC)
    h_full = np.clip(h_full, 0.0, 1.0)
    h_uint8 = (h_full * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(h_uint8, cmap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, heat_rgb, alpha, 0)
    return overlay


def _heatmap_overlay_band(img_rgb, heat_small, low=0.45, high=0.75,
                          alpha=0.5, cmap=cv2.COLORMAP_JET, suppressed=0.10,
                          global_norm=None):
    """Heatmap com filtro de banda.

    Modo de normalizacao:
        global_norm=None      -> min-max do proprio heatmap (banda relativa por painel)
        global_norm=<float>   -> divide por esse escalar (banda absoluta, comparavel
                                 entre paineis quando todos usam o mesmo global_norm)

    Apos normalizacao:
        - valores em [low, high]   -> mantem-se em [low, high] (verde -> laranja em JET)
        - valores fora da banda    -> empurrados para 'suppressed' (azul/ciano em JET)
    """
    h = heat_small.astype(np.float32)
    if global_norm is not None and global_norm > 0:
        h = h / global_norm
    else:
        rng = h.max() - h.min()
        h = (h - h.min()) / (rng + 1e-9)
    h = np.clip(h, 0.0, 1.0)
    in_band = (h >= low) & (h <= high)
    h_filt = np.where(in_band, h, suppressed).astype(np.float32)

    H, W = img_rgb.shape[:2]
    h_full = cv2.resize(h_filt, (W, H), interpolation=cv2.INTER_CUBIC)
    h_full = np.clip(h_full, 0.0, 1.0)
    h_uint8 = (h_full * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(h_uint8, cmap)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, heat_rgb, alpha, 0)
    return overlay


def _put_label(img_bgr, text):
    """Escreve um rotulo no canto superior esquerdo (com fundo preto)."""
    out = img_bgr.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (5, 5), (5 + tw + 10, 5 + th + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 5 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _entropy(p, eps=1e-12):
    """Entropia de Shannon em nats. p: vetor 1D nao-negativo (normaliza internamente)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = np.clip(p, eps, 1.0)
    return float(-(p * np.log(p)).sum())


def _concentration_metrics(attn):
    """Metricas de concentracao da atencao.

    attn: numpy [H, N, N]
    Retorna dict com:
        top1_share_per_head    : [H] fracao da massa de atencao recebida no patch top-1
                                 (max do marginal sobre queries; sum=1, entao max=share)
        top1_share_mean        : MEDIA dos top1 por head (cada head pode ter pico em
                                 posicao diferente; este valor SOMA picos individuais)
        attn_max_raw_per_head  : [H] max absoluto da matriz [N,N] de cada head
        attn_max_raw_mean      : media dos max_raw por head

        Calculados sobre a DISTRIBUICAO MEDIA (o que aparece em mean_heads.png):
        top1_of_mean_dist      : top1 da distribuicao media entre heads
                                 (pode ser MUITO menor que top1_share_mean se as heads
                                 divergem em posicao)
        max_raw_of_mean_dist   : max da matriz [N,N] obtida pela media entre heads
    """
    H, _, _ = attn.shape
    top1 = []
    max_raw = []
    for h in range(H):
        marginal = attn[h].mean(axis=0)        # [N], soma = 1
        top1.append(float(marginal.max()))
        max_raw.append(float(attn[h].max()))   # max em (queries x keys)

    # metricas da DISTRIBUICAO MEDIA (consistentes com mean_heads.png)
    avg_marginal = attn.mean(axis=0).mean(axis=0)        # [N]
    avg_matrix = attn.mean(axis=0)                       # [N, N]
    top1_of_mean_dist = float(avg_marginal.max())
    max_raw_of_mean_dist = float(avg_matrix.max())

    return {
        "top1_share_per_head": top1,
        "top1_share_mean": float(np.mean(top1)),
        "attn_max_raw_per_head": max_raw,
        "attn_max_raw_mean": float(np.mean(max_raw)),
        "top1_of_mean_dist": top1_of_mean_dist,
        "max_raw_of_mean_dist": max_raw_of_mean_dist,
    }


def _spatial_entropy(attn):
    """Entropia espacial da atencao recebida.

    attn: numpy [H, N, N] (atencao do bloco, queries x keys)
    Retorna dict com:
        per_head     : lista [H] de entropia por head (nats)
        per_head_norm: lista [H] de entropia normalizada por log(N)  (0=focal, 1=uniforme)
        mean         : media simples das entropias por head (nats)
        mean_norm    : media simples das entropias normalizadas
    """
    H, N, _ = attn.shape
    log_N = np.log(N)
    per_head = []
    per_head_norm = []
    for h in range(H):
        recv = attn[h].mean(axis=0)              # marginal por key (recebido)
        e = _entropy(recv)                       # nats
        per_head.append(e)
        per_head_norm.append(e / log_N if log_N > 0 else 0.0)
    return {
        "per_head": per_head,
        "per_head_norm": per_head_norm,
        "mean": float(np.mean(per_head)),
        "mean_norm": float(np.mean(per_head_norm)),
    }


def _list_images(folder, recursive=True):
    """Lista todas as imagens (extensoes em IMG_EXTS) na pasta."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Pasta nao existe: {folder}")
    files = []
    if recursive:
        for ext in IMG_EXTS:
            files.extend(folder.rglob(f"*{ext}"))
    else:
        for ext in IMG_EXTS:
            files.extend(folder.glob(f"*{ext}"))
    files = sorted(set(files))
    return files


def _yolo_crop_image(enc, img_path, padding=0.25, conf_threshold=0.25,
                     target_class=None, img_size=640):
    """Roda YOLO na imagem, cropa em torno da deteccao de maior confianca.

    Retorna:
        x_tensor    : [1, 3, img_size, img_size] float [0,1] no device do encoder
        crop_bgr    : numpy uint8 BGR do crop redimensionado (para overlays)
        full_bgr    : numpy uint8 BGR da imagem original
        box_info    : (x1, y1, x2, y2, cx1, cy1, cx2, cy2) em coords do original
                      onde (x1..y2) = box do YOLO, (cx1..cy2) = crop com padding
    Levanta RuntimeError se nao houver deteccao valida.
    """
    img_path = Path(img_path)
    full_bgr = cv2.imread(str(img_path))
    if full_bgr is None:
        raise FileNotFoundError(img_path)

    results = enc.yolo.predict(source=full_bgr, conf=conf_threshold,
                               verbose=False, device=str(enc.device))
    if not results:
        raise RuntimeError(f"YOLO nao retornou resultados: {img_path.name}")
    res = results[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError(f"YOLO sem deteccao em: {img_path.name}")

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    if target_class is not None:
        mask = cls == target_class
        xyxy = xyxy[mask]
        conf = conf[mask]
        if len(xyxy) == 0:
            raise RuntimeError(f"YOLO sem classe {target_class} em: {img_path.name}")

    idx = int(np.argmax(conf))
    x1, y1, x2, y2 = xyxy[idx].astype(int)

    H, W = full_bgr.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * padding))
    pad_y = int(round(bh * padding))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x)
    cy2 = min(H, y2 + pad_y)

    crop = full_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        raise RuntimeError(f"Crop invalido em: {img_path.name}")

    crop_resized = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    x_tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x_tensor = x_tensor.to(enc.device)

    return x_tensor, crop_resized, full_bgr, (x1, y1, x2, y2, cx1, cy1, cx2, cy2)


# -------------------------------------------------------------------
# Inspecao por imagem
# -------------------------------------------------------------------
def inspect_one(enc, img_path, out_dir, alpha=0.5,
                band_low=0.45, band_high=0.75, band_global=False,
                crop_with_yolo=False, crop_padding=0.25,
                crop_conf_threshold=0.25, crop_target_class=None):
    """Roda uma imagem pelo encoder, gera heatmaps e salva em out_dir.

    band_global=False (default): a banda [low, high] vale APOS min-max de cada heatmap
                                 (banda relativa, cada painel reescala).
    band_global=True           : a banda vale APOS divisao por um escalar comum a todos
                                 os paineis (banda absoluta, paineis comparaveis entre si).

    crop_with_yolo=False (default): passa a imagem inteira (resized 640x640) pelo encoder.
                                    Use com encoder treinado em quadros inteiros.
    crop_with_yolo=True            : roda YOLO, cropa em torno da deteccao com padding,
                                     e passa o crop pelo encoder.
                                     Use com encoder treinado em frames_pool_crops.
    """
    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if crop_with_yolo:
        x, orig_640, full_bgr, box_info = _yolo_crop_image(
            enc, img_path,
            padding=crop_padding,
            conf_threshold=crop_conf_threshold,
            target_class=crop_target_class,
            img_size=640,
        )
        # salva a imagem original com a box do YOLO + retangulo do crop visivel
        x1, y1, x2, y2, cx1, cy1, cx2, cy2 = box_info
        full_with_box = full_bgr.copy()
        cv2.rectangle(full_with_box, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)   # verde = crop
        cv2.rectangle(full_with_box, (x1, y1), (x2, y2), (0, 0, 255), 2)        # vermelho = box YOLO
        cv2.imwrite(str(out_dir / "00_full_with_box.png"), full_with_box)
    else:
        x = enc._read_and_resize(img_path)
        full_bgr = cv2.imread(str(img_path))
        if full_bgr is None:
            raise FileNotFoundError(img_path)
        orig_640 = cv2.resize(full_bgr, (640, 640), interpolation=cv2.INTER_LINEAR)

    # forward sem mascarar
    _tokens, attn_last = enc._forward(x)
    attn = attn_last.squeeze(0).cpu().numpy()        # [H, N, N]
    H_heads, N, _ = attn.shape
    grid = int(round(np.sqrt(N)))
    if grid * grid != N:
        raise RuntimeError(f"Grid nao-quadrado: N={N}")

    orig_rgb = cv2.cvtColor(orig_640, cv2.COLOR_BGR2RGB)

    # original
    orig_lbl = _put_label(orig_640, "Original 640x640")
    cv2.imwrite(str(out_dir / "00_original.png"), orig_lbl)
    panels = [orig_lbl]

    # metricas (calculadas antes dos overlays)
    ent = _spatial_entropy(attn)
    conc = _concentration_metrics(attn)

    # se band_global, calcula um escalar comum a todos os paineis (heads + media)
    if band_global:
        per_head_max = max(float(attn[h].mean(axis=0).max()) for h in range(H_heads))
        mean_max = float(attn.mean(axis=0).mean(axis=0).max())
        global_norm = max(per_head_max, mean_max, 1e-9)
    else:
        global_norm = None
    band_tag = "BANDA-GLOBAL" if band_global else "BANDA"

    # heatmap por head (atencao RECEBIDA = media sobre queries)
    panels_band = [orig_lbl]
    for h in range(H_heads):
        recv = attn[h].mean(axis=0).reshape(grid, grid)

        # versao normal
        ov_rgb = _heatmap_overlay(orig_rgb, recv, alpha=alpha)
        ov_bgr = cv2.cvtColor(ov_rgb, cv2.COLOR_RGB2BGR)
        lbl = (f"Head {h+1}  H={ent['per_head_norm'][h]:.2f}  "
               f"top1={conc['top1_share_per_head'][h]:.3f}")
        ov_bgr = _put_label(ov_bgr, lbl)
        cv2.imwrite(str(out_dir / f"head_{h+1}.png"), ov_bgr)
        panels.append(ov_bgr)

        # versao banda (verde-laranja realcado)
        ov_band_rgb = _heatmap_overlay_band(orig_rgb, recv,
                                            low=band_low, high=band_high,
                                            alpha=alpha,
                                            global_norm=global_norm)
        ov_band_bgr = cv2.cvtColor(ov_band_rgb, cv2.COLOR_RGB2BGR)
        ov_band_bgr = _put_label(ov_band_bgr,
                                 f"Head {h+1} {band_tag} [{band_low:.2f},{band_high:.2f}]")
        cv2.imwrite(str(out_dir / f"head_{h+1}_band.png"), ov_band_bgr)
        panels_band.append(ov_band_bgr)

    # media entre heads (versao normal) - label corrigida com top1 da distribuicao media
    mean_recv = attn.mean(axis=0).mean(axis=0).reshape(grid, grid)
    ov_mean_rgb = _heatmap_overlay(orig_rgb, mean_recv, alpha=alpha)
    ov_mean_bgr = cv2.cvtColor(ov_mean_rgb, cv2.COLOR_RGB2BGR)
    lbl_mean = (f"Media  H_norm={ent['mean_norm']:.3f}  "
                f"top1={conc['top1_of_mean_dist']:.3f}  "
                f"max_raw={conc['max_raw_of_mean_dist']:.3f}")
    ov_mean_bgr = _put_label(ov_mean_bgr, lbl_mean)
    cv2.imwrite(str(out_dir / "mean_heads.png"), ov_mean_bgr)
    panels.append(ov_mean_bgr)

    # media entre heads (versao banda)
    ov_mean_band_rgb = _heatmap_overlay_band(orig_rgb, mean_recv,
                                             low=band_low, high=band_high,
                                             alpha=alpha,
                                             global_norm=global_norm)
    ov_mean_band_bgr = cv2.cvtColor(ov_mean_band_rgb, cv2.COLOR_RGB2BGR)
    ov_mean_band_bgr = _put_label(ov_mean_band_bgr,
                                  f"Media {band_tag} [{band_low:.2f},{band_high:.2f}]")
    cv2.imwrite(str(out_dir / "mean_heads_band.png"), ov_mean_band_bgr)
    panels_band.append(ov_mean_band_bgr)

    # paineis horizontais
    panel = cv2.hconcat(panels)
    cv2.imwrite(str(out_dir / "panel.png"), panel)
    panel_band = cv2.hconcat(panels_band)
    cv2.imwrite(str(out_dir / "panel_band.png"), panel_band)

    return {
        "img": str(img_path),
        "out_dir": str(out_dir),
        "n_heads": H_heads,
        "grid": grid,
        "panel_shape": panel.shape,
        "entropy_mean_nats": ent["mean"],
        "entropy_mean_norm": ent["mean_norm"],
        "entropy_per_head_norm": ent["per_head_norm"],
        "top1_share_mean": conc["top1_share_mean"],
        "top1_share_per_head": conc["top1_share_per_head"],
        "attn_max_raw_mean": conc["attn_max_raw_mean"],
        "attn_max_raw_per_head": conc["attn_max_raw_per_head"],
        "top1_of_mean_dist": conc["top1_of_mean_dist"],
        "max_raw_of_mean_dist": conc["max_raw_of_mean_dist"],
        "global_norm_used": (global_norm if band_global else None),
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Heatmap de atencao do encoder MAE.")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--img", default=None, help="caminho de UMA imagem")
    grp.add_argument("--dir", default=None,
                     help=f"pasta com imagens (default: {DEFAULT_DIR})")
    ap.add_argument("--no-recursive", action="store_true",
                    help="(modo --dir) nao percorre subpastas")
    ap.add_argument("--yolo", default=None, help="path do .pt do YOLO (default: best.pt)")
    ap.add_argument("--mae", default=None, help="path do .pt do MAE (default: ckpt_mae/mae_epoch_020.pt)")
    ap.add_argument("--out_dir", default=None,
                    help="raiz das saidas (default: inspecao_lote\\ ao lado do script)")
    ap.add_argument("--alpha", type=float, default=0.5, help="opacidade do heatmap [0..1]")
    ap.add_argument("--band_low", type=float, default=0.45,
                    help="limite inferior da banda de realce (apos normalizacao 0..1)")
    ap.add_argument("--band_high", type=float, default=0.75,
                    help="limite superior da banda de realce (apos normalizacao 0..1)")
    ap.add_argument("--band_global", action="store_true",
                    help="banda absoluta: normaliza com escalar comum a todos os paineis "
                         "(comparacao honesta entre heads). Default e min-max por painel.")
    ap.add_argument("--crop_with_yolo", action="store_true",
                    help="cropa cada imagem na deteccao do YOLO antes de passar pelo encoder. "
                         "Use quando o encoder foi treinado em frames_pool_crops.")
    ap.add_argument("--crop_padding", type=float, default=0.25,
                    help="padding fracional ao redor da box do YOLO (default 0.25)")
    ap.add_argument("--crop_conf_threshold", type=float, default=0.25,
                    help="confianca minima do YOLO para aceitar deteccao")
    ap.add_argument("--crop_target_class", type=int, default=None,
                    help="filtra deteccoes por indice de classe (None = aceita todas)")
    args = ap.parse_args()

    # quando --crop_with_yolo e nenhum --mae explicito, usa o checkpoint dos crops
    if args.crop_with_yolo and args.mae is None and DEFAULT_MAE_PT_CROPS.exists():
        args.mae = str(DEFAULT_MAE_PT_CROPS)
        print(f"[info] --crop_with_yolo ativo: usando MAE de crops -> {args.mae}")

    # carrega encoder uma unica vez
    EncoderMAE = _import_encoder_module()
    kwargs = {}
    if args.yolo:
        kwargs["yolo_pt"] = args.yolo
    if args.mae:
        kwargs["mae_pt"] = args.mae
    enc = EncoderMAE(**kwargs)

    # decide entre 1 imagem ou pasta
    if args.img:
        img_path = Path(args.img)
        if not img_path.exists():
            print(f"[erro] imagem nao existe: {img_path}", file=sys.stderr)
            sys.exit(1)
        out_root = Path(args.out_dir) if args.out_dir else (img_path.parent / f"inspecao_{img_path.stem}")
        info = inspect_one(enc, img_path, out_root, alpha=args.alpha,
                           band_low=args.band_low, band_high=args.band_high,
                           band_global=args.band_global,
                           crop_with_yolo=args.crop_with_yolo,
                           crop_padding=args.crop_padding,
                           crop_conf_threshold=args.crop_conf_threshold,
                           crop_target_class=args.crop_target_class)
        print(f"[OK] {info['img']} -> {info['out_dir']}")
        return

    # modo pasta (default)
    folder = Path(args.dir) if args.dir else DEFAULT_DIR
    if not folder.exists():
        print(f"[erro] pasta nao existe: {folder}", file=sys.stderr)
        print(f"       crie '{folder}' e coloque suas imagens de teste la dentro.", file=sys.stderr)
        sys.exit(1)

    files = _list_images(folder, recursive=not args.no_recursive)
    if not files:
        print(f"[erro] nenhuma imagem em {folder}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(args.out_dir) if args.out_dir else (HERE / "inspecao_lote")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[info] processando {len(files)} imagens de {folder}")
    print(f"[info] saidas em: {out_root}")

    t0 = time.time()
    # colunas fixas (medias) + colunas por head (vao depender de n_heads, montadas no 1o item)
    header_fixed = [
        "src", "out_subdir", "n_heads", "grid", "panel_w", "panel_h",
        "entropy_mean_nats", "entropy_mean_norm",
        "top1_share_mean", "attn_max_raw_mean",
        "top1_of_mean_dist", "max_raw_of_mean_dist",
    ]
    summary_lines = []
    n_heads_seen = None
    for fp in tqdm(files, desc="inspecionar"):
        sub = out_root / fp.stem
        try:
            info = inspect_one(enc, fp, sub, alpha=args.alpha,
                               band_low=args.band_low, band_high=args.band_high,
                               band_global=args.band_global,
                               crop_with_yolo=args.crop_with_yolo,
                               crop_padding=args.crop_padding,
                               crop_conf_threshold=args.crop_conf_threshold,
                               crop_target_class=args.crop_target_class)
            if n_heads_seen is None:
                n_heads_seen = info["n_heads"]
                head_cols = (
                    [f"entropy_norm_h{i+1}" for i in range(n_heads_seen)] +
                    [f"top1_share_h{i+1}" for i in range(n_heads_seen)] +
                    [f"attn_max_raw_h{i+1}" for i in range(n_heads_seen)]
                )
                summary_lines.append(",".join(header_fixed + head_cols))
            row = [
                str(fp), str(sub), str(info["n_heads"]), str(info["grid"]),
                str(info["panel_shape"][1]), str(info["panel_shape"][0]),
                f"{info['entropy_mean_nats']:.6f}",
                f"{info['entropy_mean_norm']:.6f}",
                f"{info['top1_share_mean']:.6f}",
                f"{info['attn_max_raw_mean']:.6f}",
                f"{info['top1_of_mean_dist']:.6f}",
                f"{info['max_raw_of_mean_dist']:.6f}",
            ]
            row += [f"{e:.6f}" for e in info["entropy_per_head_norm"]]
            row += [f"{e:.6f}" for e in info["top1_share_per_head"]]
            row += [f"{e:.6f}" for e in info["attn_max_raw_per_head"]]
            summary_lines.append(",".join(row))
        except Exception as e:
            print(f"[skip] {fp.name}: {e}", file=sys.stderr)

    if not summary_lines:
        summary_lines = [",".join(header_fixed)]
    (out_root / "summary.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[OK] {len(files)} imagens processadas em {time.time() - t0:.1f}s")
    print(f"     resumo: {out_root / 'summary.csv'}")
    print("     entropy_mean_norm:    0=foco extremo  |  1=atencao uniforme")
    print("     top1_share_mean:      MEDIA dos picos de cada head (overestima quando heads divergem)")
    print("     top1_of_mean_dist:    pico real da distribuicao MEDIA entre heads")
    print("     max_raw_of_mean_dist: pico real da matriz [N,N] media entre heads")
    if args.band_global:
        print("     [BANDA-GLOBAL ativa: paineis comparaveis com escala absoluta]")


if __name__ == "__main__":
    main()
