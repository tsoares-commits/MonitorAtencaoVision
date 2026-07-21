"""
laplacian_topk_preview.py
-------------------------
Validacao previa da proposta de mascaramento Laplaciano-based:
para cada imagem em Imagens_Test\\, calcula resposta do Laplaciano agregada por
patch (5x5), desenha os top-6 patches em verde (visiveis) e os 19 outros em
vermelho-translucido.

Diferenca principal vs Sobel:
    - Sobel mede gradiente de PRIMEIRA ordem (transicao suave -> alto)
    - Laplaciano mede derivada de SEGUNDA ordem (zero-crossings, picos isolados)
    O Laplaciano realca CONTORNOS finos e PONTOS isolados de alta variacao;
    tende a ignorar regioes de gradiente uniforme.

Se os top-6 cairem visualmente na borda da poca de fusao na maioria das imagens,
a proposta de mascaramento determinico tem chance de funcionar.

Uso:
    python laplacian_topk_preview.py
    python laplacian_topk_preview.py --dir Imagens_Test --out_dir preview_laplacian
    python laplacian_topk_preview.py --top_k 6 --grid 5 --crop_with_yolo
    python laplacian_topk_preview.py --ksize 5 --agg mean_abs
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "Imagens_Test"
DEFAULT_OUT = HERE / "preview_laplacian"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF")


def list_images(folder, recursive=True):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(folder)
    files = []
    it = folder.rglob if recursive else folder.glob
    for ext in IMG_EXTS:
        files.extend(it(f"*{ext}"))
    return sorted(set(files))


def crop_with_yolo(img_bgr, yolo, padding=0.25, conf_threshold=0.25, target_class=None):
    results = yolo.predict(source=img_bgr, conf=conf_threshold, verbose=False)
    if not results:
        return None
    res = results[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    if target_class is not None:
        mask = cls == target_class
        xyxy = xyxy[mask]; conf = conf[mask]
        if len(xyxy) == 0:
            return None
    idx = int(np.argmax(conf))
    x1, y1, x2, y2 = xyxy[idx].astype(int)
    H, W = img_bgr.shape[:2]
    bw = max(1, x2 - x1); bh = max(1, y2 - y1)
    pad_x = int(round(bw * padding)); pad_y = int(round(bh * padding))
    cx1 = max(0, x1 - pad_x); cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x); cy2 = min(H, y2 + pad_y)
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0 or min(crop.shape[:2]) < 2:
        return None
    return crop


def laplacian_patch_scores(img_bgr, grid=5, agg="mean_sq", ksize=3):
    """Calcula score Laplaciano agregado por patch num grid GxG.

    img_bgr: HxWx3 uint8
    grid: numero de patches por lado (5 -> 5x5 = 25 patches)
    agg: 'mean_sq' (mean of squared response) ou 'mean_abs' (mean of |laplaciano|)
    ksize: tamanho do kernel Laplaciano (1, 3, 5 ou 7). Maior ksize = mais suave.

    Retorna scores [grid, grid] e mapa de magnitude HxW.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    if agg == "mean_sq":
        mag = lap * lap
    else:  # mean_abs
        mag = np.abs(lap)

    H, W = mag.shape
    # garantir divisibilidade exata redimensionando o mapa
    h_step = H // grid
    w_step = W // grid
    if h_step == 0 or w_step == 0:
        raise RuntimeError(f"Imagem pequena demais para grid={grid}: {H}x{W}")
    mag_crop = mag[:h_step * grid, :w_step * grid]
    patches = mag_crop.reshape(grid, h_step, grid, w_step).mean(axis=(1, 3))
    return patches, mag


def draw_topk_overlay(img_bgr, scores, top_k=6, alpha_keep=0.0, alpha_mask=0.55):
    """Desenha overlay: patches top-k transparentes (visiveis), restante vermelho."""
    H, W = img_bgr.shape[:2]
    grid = scores.shape[0]
    flat = scores.flatten()
    order = np.argsort(-flat)
    keep_idx = set(order[:top_k].tolist())

    h_step = H // grid
    w_step = W // grid

    overlay = img_bgr.copy()
    for r in range(grid):
        for c in range(grid):
            i = r * grid + c
            y0 = r * h_step; y1 = y0 + h_step
            x0 = c * w_step; x1 = x0 + w_step
            if i in keep_idx:
                # patch visivel: borda verde, sem overlay
                cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
            else:
                # patch mascarado: vermelho translucido
                roi = overlay[y0:y1, x0:x1]
                red = np.zeros_like(roi)
                red[..., 2] = 255
                roi[:] = cv2.addWeighted(roi, 1 - alpha_mask, red, alpha_mask, 0)
                cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 0, 120), 1)

    # adiciona ranking nos cantos dos top-k
    rank = {idx: pos for pos, idx in enumerate(order[:top_k].tolist())}
    for i, pos in rank.items():
        r, c = i // grid, i % grid
        x = c * w_step + 8
        y = r * h_step + 24
        cv2.putText(overlay, f"#{pos+1}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return overlay


def main():
    ap = argparse.ArgumentParser(description="Visualiza top-K patches por resposta do Laplaciano.")
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--grid", type=int, default=5, help="GxG patches (5 = 25 tokens, igual ao MAE)")
    ap.add_argument("--top_k", type=int, default=6,
                    help="quantos patches marcar como visiveis (6 ~ 25%% de 25)")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--crop_with_yolo", action="store_true",
                    help="cropa cada imagem na box do YOLO antes do Laplaciano (mesmo dominio do treino)")
    ap.add_argument("--yolo", default=str(HERE / "best.pt"))
    ap.add_argument("--crop_padding", type=float, default=0.25)
    ap.add_argument("--crop_target_class", type=int, default=None)
    ap.add_argument("--agg", default="mean_sq", choices=["mean_sq", "mean_abs"])
    ap.add_argument("--ksize", type=int, default=3, choices=[1, 3, 5, 7],
                    help="tamanho do kernel Laplaciano (1=ruidoso, 3=padrao, 5/7=mais suave)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_images(args.dir, recursive=True)
    if not files:
        print(f"[erro] sem imagens em {args.dir}")
        return

    yolo = None
    if args.crop_with_yolo:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)

    summary = [["src", "out", "topk_positions", "topk_scores"]]
    for fp in tqdm(files, desc="laplacian-preview"):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        if args.crop_with_yolo:
            cropped = crop_with_yolo(img, yolo,
                                     padding=args.crop_padding,
                                     conf_threshold=0.25,
                                     target_class=args.crop_target_class)
            if cropped is None:
                continue
            img = cropped
        img640 = cv2.resize(img, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

        scores, mag = laplacian_patch_scores(img640, grid=args.grid,
                                             agg=args.agg, ksize=args.ksize)
        overlay = draw_topk_overlay(img640, scores, top_k=args.top_k)

        # heatmap auxiliar (Laplaciano renormalizado)
        mag_n = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
        mag_u8 = (mag_n * 255).astype(np.uint8)
        heat = cv2.applyColorMap(mag_u8, cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (img640.shape[1], img640.shape[0]))
        side_by_side = cv2.hconcat([img640, heat, overlay])

        out_name = f"{fp.stem}_topk{args.top_k}.png"
        cv2.imwrite(str(out_dir / out_name), side_by_side)

        flat = scores.flatten()
        order = np.argsort(-flat)
        topk_idx = order[:args.top_k].tolist()
        topk_scr = flat[order[:args.top_k]].tolist()
        positions = [(int(i // args.grid), int(i % args.grid)) for i in topk_idx]
        summary.append([str(fp), str(out_dir / out_name), str(positions),
                        str([f"{s:.1f}" for s in topk_scr])])

    csv_path = out_dir / "_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary)

    print(f"\n[OK] saidas em {out_dir}")
    print( "     cada imagem: original | mapa Laplaciano | original com top-K marcados")
    print(f"     csv: {csv_path}")


if __name__ == "__main__":
    main()