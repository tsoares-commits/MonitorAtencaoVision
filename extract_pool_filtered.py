"""
extract_pool_filtered.py
------------------------
Filtra frames pela DETECCAO do YOLO (nao cropa). Para cada frame em FRAMES_DIR,
roda o YOLO e SE houver deteccao da poca (qualquer classe ou classe alvo),
copia o frame INTEIRO para a pasta de saida.

Diferenca de extract_pool_crops.py:
    - extract_pool_crops.py:    YOLO detecta -> CROPA box -> resize -> salva crop
    - extract_pool_filtered.py: YOLO detecta -> SALVA FRAME INTEIRO (sem cropar)

Uso:
    python extract_pool_filtered.py
    python extract_pool_filtered.py --target_class 0 --conf_threshold 0.30
    python extract_pool_filtered.py --frames_dir "outra_pasta" --out_dir "outra_saida"

Saida:
    <out_dir>/<nome_do_frame>.jpg  (frame completo, redimensionado p/ img_size)
    <out_dir>/_filter_log.csv      (quais frames foram aceitos/pulados e motivos)
"""

import argparse
import csv
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Reusa caminhos do treino
from train_mae_local import (
    YOLO_WEIGHTS, IMG_SIZE,
)


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "frames_train_vision_5642_frames"
DEFAULT_OUT = HERE / "frames_pool_filtered"

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


def main():
    ap = argparse.ArgumentParser(description="Filtra frames pelas deteccoes do YOLO (sem crop).")
    ap.add_argument("--frames_dir", default=str(DEFAULT_INPUT),
                    help="pasta com frames originais")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT),
                    help="pasta de saida (frames inteiros filtrados)")
    ap.add_argument("--yolo", default=str(YOLO_WEIGHTS),
                    help="path do .pt do YOLO")
    ap.add_argument("--target_class", type=int, default=None,
                    help="indice de classe a filtrar (None = aceita qualquer)")
    ap.add_argument("--conf_threshold", type=float, default=0.25,
                    help="confianca minima do YOLO")
    ap.add_argument("--img_size", type=int, default=IMG_SIZE,
                    help="tamanho final do frame quadrado salvo (default: IMG_SIZE)")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--device", default=None,
                    help="device do YOLO (cpu, cuda); default = auto")
    args = ap.parse_args()

    from ultralytics import YOLO

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[filter] frames_dir   = {args.frames_dir}")
    print(f"[filter] out_dir      = {out_dir}")
    print(f"[filter] yolo         = {args.yolo}")
    print(f"[filter] target_class = {args.target_class}")
    print(f"[filter] conf_thresh  = {args.conf_threshold}")
    print(f"[filter] img_size     = {args.img_size}")

    if not Path(args.yolo).exists():
        raise FileNotFoundError(f"YOLO weights nao encontrado: {args.yolo}")

    yolo = YOLO(args.yolo)

    files = list_images(args.frames_dir, recursive=not args.no_recursive)
    if not files:
        raise FileNotFoundError(f"sem imagens em {args.frames_dir}")
    print(f"[filter] processando {len(files)} frames")

    log_rows = []
    n_kept = 0
    n_skipped = 0
    t0 = time.time()

    for fp in tqdm(files, desc="filtering"):
        img = cv2.imread(str(fp))
        if img is None:
            log_rows.append([str(fp), 0, 0, "fail_read"])
            n_skipped += 1
            continue

        try:
            results = yolo.predict(source=img, conf=args.conf_threshold,
                                   verbose=False, device=args.device)
        except Exception as e:
            log_rows.append([str(fp), 0, 0, f"yolo_err:{e}"])
            n_skipped += 1
            continue

        if not results:
            log_rows.append([str(fp), 0, 0, "no_results"])
            n_skipped += 1
            continue

        res = results[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            log_rows.append([str(fp), 0, 0, "no_detection"])
            n_skipped += 1
            continue

        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()

        if args.target_class is not None:
            mask = cls == args.target_class
            if not mask.any():
                log_rows.append([str(fp), len(boxes), 0, f"no_class_{args.target_class}"])
                n_skipped += 1
                continue
            best_conf = float(conf[mask].max())
            n_target = int(mask.sum())
        else:
            best_conf = float(conf.max())
            n_target = len(boxes)

        # salva frame INTEIRO redimensionado (sem cropar pela box)
        img_resized = cv2.resize(img, (args.img_size, args.img_size),
                                 interpolation=cv2.INTER_LINEAR)
        out_path = out_dir / f"{fp.stem}.jpg"
        cv2.imwrite(str(out_path), img_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

        log_rows.append([str(fp), len(boxes), n_target, f"kept_conf={best_conf:.2f}"])
        n_kept += 1

    # log CSV
    csv_path = out_dir / "_filter_log.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_frame", "n_detections_total", "n_detections_kept", "status"])
        w.writerows(log_rows)

    dt = time.time() - t0
    print(f"\n[OK] {dt:.1f}s")
    print(f"     frames mantidos: {n_kept}")
    print(f"     frames pulados : {n_skipped}")
    print(f"     log: {csv_path}")
    print()
    print("Proximos passos:")
    print(f"  1) Em train_mae_local.py confira FRAMES_DIR = HERE / \"{out_dir.name}\"")
    print( "  2) python train_mae_local.py extract")
    print( "  3) python train_mae_local.py train")


if __name__ == "__main__":
    main()