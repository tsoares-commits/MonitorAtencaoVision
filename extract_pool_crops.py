"""
extract_pool_crops.py
---------------------
Para cada frame em FRAMES_DIR, roda o YOLO, pega a(s) caixa(s) de deteccao
(opcionalmente filtrando por classe), aplica padding ao redor da caixa, cropa
e redimensiona para IMG_SIZE x IMG_SIZE. Salva em uma pasta nova.

Saida: <out_dir>/*.png + <out_dir>/_extract_log.csv

Uso:
    python extract_pool_crops.py
    python extract_pool_crops.py --target_class 0 --padding 0.30
    python extract_pool_crops.py --max_per_frame 3 --conf_threshold 0.30
    python extract_pool_crops.py --frames_dir "outra_pasta" --out_dir "crops_v2"

Depois de gerar, ajustar:
    train_mae_local.py  ->  FRAMES_DIR = HERE / "frames_pool_crops"
e rodar:
    python train_mae_local.py extract     # gera novo cache_yolo_feats/
    python train_mae_local.py train       # treina MAE em cima dos crops
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Reusa caminhos do treino para manter consistencia
from train_mae_local import (
    YOLO_WEIGHTS, FRAMES_DIR, IMG_SIZE,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "frames_pool_crops"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF")


def list_images(folder, recursive=True):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Pasta nao existe: {folder}")
    files = []
    it = folder.rglob if recursive else folder.glob
    for ext in IMG_EXTS:
        files.extend(it(f"*{ext}"))
    return sorted(set(files))


def crop_with_padding(img, x1, y1, x2, y2, pad_frac):
    """Cropa img com pad_frac extra em cada lado, com clamping nas bordas.

    Retorna (crop, (cx1, cy1, cx2, cy2)).
    """
    H, W = img.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x)
    cy2 = min(H, y2 + pad_y)
    return img[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


def main():
    ap = argparse.ArgumentParser(description="Cropa frames em torno das deteccoes do YOLO.")
    ap.add_argument("--frames_dir", default=str(FRAMES_DIR),
                    help="pasta com frames originais")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT),
                    help="pasta de saida dos crops")
    ap.add_argument("--yolo", default=str(YOLO_WEIGHTS),
                    help="path do .pt do YOLO")
    ap.add_argument("--padding", type=float, default=0.25,
                    help="fracao da dimensao da caixa adicionada como margem (cada lado)")
    ap.add_argument("--target_class", type=int, default=None,
                    help="indice de classe a filtrar (None = aceita todas)")
    ap.add_argument("--min_box_size", type=int, default=64,
                    help="lado minimo da caixa em pixels (descarta menores)")
    ap.add_argument("--img_size", type=int, default=IMG_SIZE,
                    help="tamanho final do crop quadrado (default: IMG_SIZE)")
    ap.add_argument("--max_per_frame", type=int, default=1,
                    help="quantos crops salvar por frame (ordena por confianca desc)")
    ap.add_argument("--conf_threshold", type=float, default=0.25,
                    help="confianca minima do YOLO para aceitar deteccao")
    ap.add_argument("--no-recursive", action="store_true",
                    help="(modo --frames_dir) nao percorre subpastas")
    ap.add_argument("--device", default=None,
                    help="device do YOLO (cpu, cuda, etc); default = auto")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[crops] frames_dir   = {args.frames_dir}")
    print(f"[crops] out_dir      = {out_dir}")
    print(f"[crops] yolo         = {args.yolo}")
    print(f"[crops] padding      = {args.padding}")
    print(f"[crops] target_class = {args.target_class}")
    print(f"[crops] min_box_size = {args.min_box_size}px")
    print(f"[crops] max_per_frame= {args.max_per_frame}")
    print(f"[crops] conf_thresh  = {args.conf_threshold}")
    print(f"[crops] img_size     = {args.img_size}")

    if not Path(args.yolo).exists():
        raise FileNotFoundError(f"YOLO weights nao encontrado: {args.yolo}")

    yolo = YOLO(args.yolo)

    files = list_images(args.frames_dir, recursive=not args.no_recursive)
    if not files:
        raise FileNotFoundError(f"nenhuma imagem em {args.frames_dir}")
    print(f"[crops] processando {len(files)} frames")

    log_rows = []
    n_with_crop = 0
    n_skipped = 0
    total_crops = 0
    t0 = time.time()

    for fp in tqdm(files, desc="cropping"):
        img = cv2.imread(str(fp))
        if img is None:
            log_rows.append([str(fp), 0, 0, "", "fail_read"])
            n_skipped += 1
            continue

        try:
            results = yolo.predict(source=img, conf=args.conf_threshold,
                                   verbose=False, device=args.device)
        except Exception as e:
            log_rows.append([str(fp), 0, 0, "", f"yolo_err:{e}"])
            n_skipped += 1
            continue

        if not results:
            log_rows.append([str(fp), 0, 0, "", "no_results"])
            n_skipped += 1
            continue

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            log_rows.append([str(fp), 0, 0, "", "no_detection"])
            n_skipped += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        # filtra por classe
        if args.target_class is not None:
            mask = cls == args.target_class
            xyxy = xyxy[mask]
            cls = cls[mask]
            conf = conf[mask]
            if len(xyxy) == 0:
                log_rows.append([str(fp), 0, 0, "", f"no_class_{args.target_class}"])
                n_skipped += 1
                continue

        # ordena por conf desc, mantem ate max_per_frame
        order = np.argsort(-conf)
        xyxy = xyxy[order][:args.max_per_frame]
        cls = cls[order][:args.max_per_frame]
        conf = conf[order][:args.max_per_frame]

        kept = 0
        crop_names = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            bw = x2 - x1
            bh = y2 - y1
            if min(bw, bh) < args.min_box_size:
                continue

            crop, _ = crop_with_padding(img, x1, y1, x2, y2, args.padding)
            if crop.size == 0 or min(crop.shape[:2]) < 2:
                continue
            crop_resized = cv2.resize(crop, (args.img_size, args.img_size),
                                      interpolation=cv2.INTER_LINEAR)

            # nome unico: <stem>__det<i>__c<class>__conf<XX>.png
            out_name = f"{fp.stem}__det{i}__c{int(cls[i])}__conf{conf[i]:.2f}.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), crop_resized)
            crop_names.append(out_name)
            kept += 1

        if kept == 0:
            log_rows.append([str(fp), len(boxes), 0, "", "all_below_min_size"])
            n_skipped += 1
        else:
            log_rows.append([str(fp), len(boxes), kept, ";".join(crop_names), ""])
            n_with_crop += 1
            total_crops += kept

    # log CSV
    csv_path = out_dir / "_extract_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_frame", "n_detections_total", "n_crops_saved",
                    "crop_paths", "reason_skipped"])
        w.writerows(log_rows)

    dt = time.time() - t0
    print(f"\n[OK] {dt:.1f}s | crops gerados: {total_crops}")
    print(f"     frames com >=1 crop: {n_with_crop}")
    print(f"     frames pulados:      {n_skipped}")
    print(f"     log: {csv_path}")
    print()
    print("Proximos passos:")
    print(f"  1) Em train_mae_local.py:  FRAMES_DIR = HERE / \"{out_dir.name}\"")
    print( "  2) python train_mae_local.py extract")
    print( "  3) python train_mae_local.py train")


if __name__ == "__main__":
    main()