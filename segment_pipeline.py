"""
segment_pipeline.py
-------------------
Pipeline: YOLO + Encoder MAE (atencao) + SAM (segmentacao do contorno
da poca dentro da box) + tracking de variacao de area no tempo.

Logica de decisao por frame (UNICO caso que segmenta):
    YOLO   ATENCAO FORTE   |  acao
    -----  -------------   |  -----------------------------------------
    SIM    SIM             |  SAM segmenta o objeto dentro da box (prompt = bbox)
    NAO    SIM             |  pula (so atencao nao basta)
    SIM    NAO             |  pula (atencao nao confiavel)
    NAO    NAO             |  pula (nada para segmentar)

A mascara e o CONTORNO REAL da poca (SAM com bbox como prompt).

Tracking de area:
    A0 = primeira area segmentada com sucesso
    ratio_t = A_t / A0  (variacao em torno do baseline)

Uso:
    # pasta de imagens (default: Imagens_Test)
    python segment_pipeline.py
    python segment_pipeline.py --dir caminho\\para\\imagens
    python segment_pipeline.py --dir Imagens_Test --out_dir resultados

    # video
    python segment_pipeline.py --video caminho\\video.mp4
    python segment_pipeline.py --video video.mp4 --out_dir resultados_video

Saidas:
    <out_dir>/<nome>_seg.png             frame anotado (mascara verde + box + label)
    <out_dir>/metrics.csv                metricas por frame (decisao, area, ratio)
    <out_dir>/<video>_segmented.mp4      video reconstruido (modo --video)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Reaproveita o pipeline ja construido
from train_mae_local import IMG_SIZE
from Aplicacao_Enconder import EncoderMAE


HERE = Path(__file__).resolve().parent
DEFAULT_DIR  = HERE / "Imagens_Test"
DEFAULT_OUT  = HERE / "segmentation_output"
DEFAULT_SAM  = HERE / "sam_b.pt"   # baixado automaticamente pela ultralytics se ausente

# Thresholds (ajustaveis via CLI)
DEFAULT_ATTN_TOP1   = 0.15   # top1 minimo p/ considerar uma head "ativa"
DEFAULT_YOLO_CONF   = 0.25
DEFAULT_TARGET_CLS  = None   # None = aceita qualquer classe
DEFAULT_FRAME_STEP  = 10     # processa 1 a cada N frames (modo video)
OVERSHOOT_RATIO     = 1.20   # area > 1.2 * A0 conta como "sobrecarga" (poça grande demais)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
            ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF")

# Cores BGR p/ anotacao
COLOR_BOX        = (0,   0, 255)   # vermelho   (box do YOLO)
COLOR_MASK_FILL  = (0, 255,   0)   # verde      (preenchimento mascara)
COLOR_MASK_LINE  = (0, 200,   0)   # verde escuro (contorno mascara)
COLOR_DECISION = {
    "BOX+ATTN":     (0, 255,   0),   # verde  - unico caso que segmenta
    "SKIP_NO_BOX":  (0, 255, 255),   # amarelo - so atencao, sem box do YOLO
    "SKIP_NO_ATTN": (0,   0, 255),   # vermelho - YOLO sem atencao
    "SKIP_NO_BOTH": (128,128, 128),  # cinza   - sem nada
}


# -----------------------------------------------------------------------------
# Helpers de I/O
# -----------------------------------------------------------------------------
def list_images(folder, recursive=True):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Pasta nao existe: {folder}")
    files = []
    it = folder.rglob if recursive else folder.glob
    for ext in IMG_EXTS:
        files.extend(it(f"*{ext}"))
    return sorted(set(files))


# -----------------------------------------------------------------------------
# YOLO: detecta box (None se nao detectar)
# -----------------------------------------------------------------------------
def get_yolo_box(yolo, img_bgr, conf_threshold=0.25, target_class=None):
    """Retorna (x1, y1, x2, y2) da deteccao de maior confianca, ou None."""
    try:
        results = yolo.predict(source=img_bgr, conf=conf_threshold, verbose=False)
    except Exception:
        return None
    if not results or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    if target_class is not None:
        mask = cls == target_class
        if not mask.any():
            return None
        xyxy = xyxy[mask]; conf = conf[mask]

    idx = int(np.argmax(conf))
    return tuple(int(v) for v in xyxy[idx])


# -----------------------------------------------------------------------------
# Encoder MAE: atenção do ultimo bloco
# -----------------------------------------------------------------------------
def get_attention_info(enc, img_bgr):
    """Roda o encoder e devolve dict com:
        top1_per_head : lista [H] de top1_share por head
        max_top1      : maximo entre as heads
        best_head     : indice da head com maior top1
        peak_xy_640   : (x, y) do pico da melhor head em coords 640x640
        attn          : tensor numpy [H, N, N] caso queira inspecionar
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(enc.device) / 255.0

    with torch.no_grad():
        _tokens, attn_last = enc._forward(x)
    attn = attn_last.squeeze(0).cpu().numpy()       # [H, N, N]
    H, N, _ = attn.shape
    grid = int(round(np.sqrt(N)))

    top1_per_head = []
    for h in range(H):
        marginal = attn[h].mean(axis=0)              # [N]
        top1_per_head.append(float(marginal.max()))

    max_top1 = float(max(top1_per_head))
    best_head = int(np.argmax(top1_per_head))

    # pico (em coords 640x640) da melhor head
    best_marginal = attn[best_head].mean(axis=0).reshape(grid, grid)
    py, px = np.unravel_index(int(np.argmax(best_marginal)), best_marginal.shape)
    h_step = IMG_SIZE // grid
    w_step = IMG_SIZE // grid
    peak_x = int((px + 0.5) * w_step)
    peak_y = int((py + 0.5) * h_step)

    return {
        "top1_per_head": top1_per_head,
        "max_top1": max_top1,
        "best_head": best_head,
        "peak_xy_640": (peak_x, peak_y),
        "attn": attn,
    }


# -----------------------------------------------------------------------------
# SAM: segmenta o CONTORNO REAL do objeto ESTRITAMENTE dentro da box do YOLO
# -----------------------------------------------------------------------------
def segment_with_sam(sam, img_bgr, box):
    """SAM com bbox como prompt -> mascara booleana [H, W] do contorno real
    do objeto, ESTRITAMENTE dentro da box do YOLO (sem expansao/padding).

    Apos receber a mascara do SAM, zera tudo fora dos limites da box original.
    Assim o pipeline trabalha apenas com o que a rede YOLO detectou.

    box: (x1, y1, x2, y2) em coords da imagem original
    """
    if box is None:
        return None

    H, W = img_bgr.shape[:2]

    try:
        results = sam(img_bgr, bboxes=[list(box)], verbose=False)
    except Exception as e:
        print(f"[SAM] erro: {e}", file=sys.stderr)
        return None

    if not results or results[0].masks is None:
        return None
    masks = results[0].masks.data.cpu().numpy()      # [N, H, W]
    if len(masks) == 0:
        return None

    # pega a mascara de maior area
    areas = [int(m.sum()) for m in masks]
    idx = int(np.argmax(areas))
    mask = masks[idx].astype(bool)

    # CLIPPING: zera tudo fora da box original (mantem mascara dentro do retangulo do YOLO)
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(W, int(x2)); y2 = min(H, int(y2))
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped


# -----------------------------------------------------------------------------
# Anotacao do frame: overlay de mascara + box + label
# -----------------------------------------------------------------------------
def annotate_frame(img_bgr, mask=None, box=None, decision="?", info_text=""):
    out = img_bgr.copy()
    if mask is not None and mask.shape[:2] == img_bgr.shape[:2]:
        # overlay verde translucido onde mask = True
        overlay = out.copy()
        overlay[mask] = COLOR_MASK_FILL
        out = cv2.addWeighted(out, 0.55, overlay, 0.45, 0)
        # contorno
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, COLOR_MASK_LINE, 2)

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_BOX, 2)

    color = COLOR_DECISION.get(decision, (255, 255, 255))
    label = decision if not info_text else f"{decision} | {info_text}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (5, 5), (5 + tw + 10, 5 + th + 12), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 5 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


# -----------------------------------------------------------------------------
# Processamento de UM frame: aplica a logica de decisao + segmentacao
# -----------------------------------------------------------------------------
def process_one_frame(img_bgr, yolo, encoder, sam,
                      attn_threshold=0.15, conf_threshold=0.25,
                      target_class=None):
    H, W = img_bgr.shape[:2]

    # 1. YOLO em resolucao original
    yolo_box = get_yolo_box(yolo, img_bgr, conf_threshold, target_class)
    has_yolo = yolo_box is not None

    # 2. Encoder MAE (atencao em coords 640x640)
    attn_info = get_attention_info(encoder, img_bgr)
    has_attn = attn_info["max_top1"] > attn_threshold

    # 3. Logica de decisao: UNICO caso que segmenta = YOLO + ATTN
    decision = "?"
    mask = None
    used_box = yolo_box   # mostra box no overlay mesmo quando pula

    if has_yolo and has_attn:
        decision = "BOX+ATTN"
        # SAM segmenta o CONTORNO REAL do objeto, estritamente dentro da box do YOLO
        # (sem padding, sem expansao). A funcao ja faz clipping na box.
        mask = segment_with_sam(sam, img_bgr, box=yolo_box)
    elif has_attn and not has_yolo:
        decision = "SKIP_NO_BOX"
    elif has_yolo and not has_attn:
        decision = "SKIP_NO_ATTN"
    else:
        decision = "SKIP_NO_BOTH"

    return {
        "decision": decision,
        "mask": mask,
        "box": used_box,
        "max_top1": attn_info["max_top1"],
        "top1_per_head": attn_info["top1_per_head"],
        "best_head": attn_info["best_head"],
    }


# -----------------------------------------------------------------------------
# Modo PASTA: processa uma colecao de imagens
# -----------------------------------------------------------------------------
def process_folder(folder, out_dir, yolo, encoder, sam,
                   attn_threshold, conf_threshold, target_class):
    files = list_images(folder)
    if not files:
        raise FileNotFoundError(f"sem imagens em {folder}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = [["frame_idx", "src", "decision", "max_top1",
                 "yolo_box", "area_px", "area_ratio_A0"]]
    A0 = None
    n_seg = 0

    for idx, fp in enumerate(tqdm(files, desc="folder")):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        r = process_one_frame(img, yolo, encoder, sam,
                              attn_threshold, conf_threshold, target_class)
        area = int(r["mask"].sum()) if r["mask"] is not None else 0
        if area > 0:
            if A0 is None:
                A0 = area
            ratio = area / A0
            n_seg += 1
        else:
            ratio = 0.0

        annotated = annotate_frame(
            img, mask=r["mask"], box=r["box"], decision=r["decision"],
            info_text=f"top1={r['max_top1']:.3f} area={area} ratio={ratio:.2f}",
        )
        out_name = f"{idx:06d}_{fp.stem}_seg.png"
        cv2.imwrite(str(out_dir / out_name), annotated)

        csv_rows.append([
            idx, str(fp), r["decision"], f"{r['max_top1']:.4f}",
            str(r["box"]) if r["box"] else "",
            area, f"{ratio:.4f}",
        ])

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\n[OK] {len(files)} imagens processadas")
    print(f"     segmentadas: {n_seg}")
    print(f"     A0 (baseline area): {A0}")
    print(f"     metricas: {csv_path}")
    print(f"     anotadas:  {out_dir}")


# -----------------------------------------------------------------------------
# Modo VIDEO: processa 1 a cada frame_step frames, agrega metricas para MLP
# -----------------------------------------------------------------------------
def process_video(video_path, out_dir, yolo, encoder, sam,
                  attn_threshold, conf_threshold, target_class,
                  frame_step=DEFAULT_FRAME_STEP, save_video=False):
    """Processa video frame-a-frame (com salto de frame_step) e produz:
        - <video_stem>_metrics.csv         agregado MLP-ready (1 linha)
        - <video_stem>_frames.csv          metricas por frame processado
        - <video_stem>/<idx>_seg.png       frames anotados (so os segmentados c/ mascara)
        - <video_stem>_segmented.mp4       so se save_video=True
    """
    # tolera espaco extra no inicio/fim do caminho (CLI no PowerShell)
    video_path = Path(str(video_path).strip().strip('"').strip("'"))
    if not video_path.exists():
        raise FileNotFoundError(f"video nao existe: '{video_path}' "
                                f"(confira espacos extras ou caminho)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"nao foi possivel abrir video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # video output (opcional)
    writer = None
    out_video_path = None
    if save_video:
        out_video_path = out_dir / f"{vid}_segmented.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"nao foi possivel criar video saida: {out_video_path}")

    # CSV por frame (debug)
    frames_csv_path = out_dir / f"{vid}_frames.csv"
    frames_f = open(frames_csv_path, "w", newline="", encoding="utf-8")
    frames_w = csv.writer(frames_f)
    frames_w.writerow(["frame_idx", "decision", "max_top1",
                       "yolo_box", "area_px", "area_ratio_A0"])

    # subpasta para frames anotados (criada apenas se houver pelo menos 1 segmentado)
    seg_frames_dir = out_dir / vid
    seg_frames_dir.mkdir(parents=True, exist_ok=True)

    A0 = None
    n_seg = 0
    idx = 0
    processed = 0
    areas = []   # acumula areas dos frames segmentados (para metricas agregadas)
    pbar = tqdm(total=n_frames_total if n_frames_total > 0 else None, desc=f"video {vid}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # so processa 1 a cada frame_step frames
            if idx % frame_step != 0:
                idx += 1
                pbar.update(1)
                continue

            r = process_one_frame(frame, yolo, encoder, sam,
                                  attn_threshold, conf_threshold, target_class)
            area = int(r["mask"].sum()) if r["mask"] is not None else 0
            if area > 0:
                if A0 is None:
                    A0 = area
                ratio = area / A0
                n_seg += 1
                areas.append(area)
            else:
                ratio = 0.0

            frames_w.writerow([
                idx, r["decision"], f"{r['max_top1']:.4f}",
                str(r["box"]) if r["box"] else "",
                area, f"{ratio:.4f}",
            ])

            # salva o frame anotado SE houve segmentacao valida (mask nao-vazia)
            annotated = None
            if r["mask"] is not None and area > 0:
                annotated = annotate_frame(
                    frame, mask=r["mask"], box=r["box"], decision=r["decision"],
                    info_text=f"top1={r['max_top1']:.3f} area={area} ratio={ratio:.2f}",
                )
                out_png = seg_frames_dir / f"{idx:06d}_seg.png"
                cv2.imwrite(str(out_png), annotated)

            if save_video:
                # para reconstrucao do video, anota TODOS os frames processados
                # (mesmo os SKIP, para o video manter a duracao original)
                if annotated is None:
                    annotated = annotate_frame(
                        frame, mask=r["mask"], box=r["box"], decision=r["decision"],
                        info_text=f"top1={r['max_top1']:.3f} area={area} ratio={ratio:.2f}",
                    )
                writer.write(annotated)

            processed += 1
            idx += 1
            pbar.update(1)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        frames_f.close()
        pbar.close()

    # === metricas agregadas para o MLP ===
    if len(areas) > 0:
        areas_np = np.array(areas, dtype=np.float64)
        area_mean = float(areas_np.mean())
        area_std = float(areas_np.std())
        area_amplitude = float(areas_np.max() - areas_np.min())
        # fracao de frames com sobrecarga (area > OVERSHOOT_RATIO * A0)
        frac_overshoot = float((areas_np > OVERSHOOT_RATIO * A0).sum() / len(areas_np))
    else:
        area_mean = area_std = area_amplitude = frac_overshoot = 0.0

    metrics_csv = out_dir / f"{vid}_metrics.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "video_id", "n_frames_total", "n_frames_processed", "n_segmented",
            "frame_step", "A0",
            "area_mean", "area_std", "area_amplitude", "frac_overshoot",
        ])
        w.writerow([
            vid, n_frames_total, processed, n_seg,
            frame_step, A0 if A0 is not None else 0,
            f"{area_mean:.2f}", f"{area_std:.2f}",
            f"{area_amplitude:.2f}", f"{frac_overshoot:.4f}",
        ])

    print(f"\n[OK] video '{vid}': {idx} frames totais | {processed} processados (1/{frame_step}) | {n_seg} segmentados")
    print(f"     A0 (baseline)   : {A0}")
    print(f"     area_mean       : {area_mean:.2f}")
    print(f"     area_std        : {area_std:.2f}  (volatilidade)")
    print(f"     area_amplitude  : {area_amplitude:.2f}")
    print(f"     frac_overshoot  : {frac_overshoot:.4f}  (frames com area > {OVERSHOOT_RATIO:.2f} * A0)")
    print(f"     metricas agreg.: {metrics_csv}")
    print(f"     metricas frames: {frames_csv_path}")
    print(f"     frames segm.   : {seg_frames_dir} ({n_seg} arquivos)")
    if save_video:
        print(f"     video anotado  : {out_video_path}")


# -----------------------------------------------------------------------------
# Main / CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="YOLO + MAE attention + SAM segmentation pipeline.")

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--dir", default=str(DEFAULT_DIR),
                     help=f"pasta com imagens (default: {DEFAULT_DIR})")
    grp.add_argument("--video", default=None, help="caminho do video (.mp4, .avi, etc.)")

    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--yolo", default=None, help="path do best.pt (default: do encoder)")
    ap.add_argument("--mae", default=None, help="path do .pt do MAE (default: do encoder)")
    ap.add_argument("--sam", default=str(DEFAULT_SAM),
                    help="path do SAM .pt (auto-baixa sam_b.pt se ausente)")

    ap.add_argument("--attn_threshold", type=float, default=DEFAULT_ATTN_TOP1,
                    help=f"top1 minimo p/ considerar atencao 'forte' (default: {DEFAULT_ATTN_TOP1})")
    ap.add_argument("--conf_threshold", type=float, default=DEFAULT_YOLO_CONF,
                    help=f"confianca minima YOLO (default: {DEFAULT_YOLO_CONF})")
    ap.add_argument("--target_class", type=int, default=DEFAULT_TARGET_CLS,
                    help="filtra deteccoes por classe (default: aceita todas)")
    ap.add_argument("--frame_step", type=int, default=DEFAULT_FRAME_STEP,
                    help=f"processa 1 a cada N frames (default: {DEFAULT_FRAME_STEP}); "
                         f"so vale no modo --video")
    ap.add_argument("--save_video", action="store_true",
                    help="reconstroi video anotado (default OFF, so gera CSV de metricas)")
    args = ap.parse_args()

    # === Carrega encoder (que ja tem YOLO embutido) ===
    print("[init] carregando YOLO + MAE encoder...")
    enc_kwargs = {}
    if args.yolo:
        enc_kwargs["yolo_pt"] = args.yolo
    if args.mae:
        enc_kwargs["mae_pt"] = args.mae
    encoder = EncoderMAE(**enc_kwargs)
    yolo = encoder.yolo

    # === Carrega SAM ===
    print(f"[init] carregando SAM ({args.sam})...")
    from ultralytics import SAM
    sam_pt = args.sam
    if not Path(sam_pt).exists():
        print("[init] SAM weights nao encontrado localmente; ultralytics vai baixar 'sam_b.pt'")
        sam_pt = "sam_b.pt"
    sam = SAM(sam_pt)
    print("[init] OK\n")

    # === Despacha modo ===
    t0 = time.time()
    if args.video:
        process_video(args.video, args.out_dir, yolo, encoder, sam,
                      args.attn_threshold, args.conf_threshold, args.target_class,
                      frame_step=args.frame_step, save_video=args.save_video)
    else:
        process_folder(args.dir, args.out_dir, yolo, encoder, sam,
                       args.attn_threshold, args.conf_threshold, args.target_class)
    print(f"[total] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()