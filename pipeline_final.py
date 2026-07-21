r"""
pipeline_final.py
-----------------
Pipeline integrado FINAL: YOLO + Vision Encoder + SAM + MLP.

Fluxo:
    1. Le a planilha "Teste MLP.xlsx" e localiza a linha da Amostra
       correspondente ao stem do video (ex: video '16.avi' -> Amostra 16).
    2. Abre o video e processa de N em N frames (default: 5):
         a) YOLO detecta a box
         b) Vision encoder MAE calcula atencao
         c) SAM segmenta DENTRO da box (regra: BOX+ATTN forte) e produz a area
         d) Atualiza metricas cumulativas (area_mean, std, amplitude, frac_overshoot, A0)
         e) Roda o MLP com [parametros do Excel] + [metricas acumuladas]
         f) Anota o frame: mascara verde + box + label + previsao MLP
    3. Salva tudo em "Resultados Finais/<id>/":
         - <id>_inputs.csv             linha do Excel usada
         - <id>_metrics_realtime.csv   metricas cumulativas por frame processado
         - <id>_mlp_predictions.csv    previsao MLP a cada frame processado
         - <id>_final_summary.json     ultimo predict + estatistica final
         - <id>_reconstructed.mp4      video anotado com overlay

Uso:
    python pipeline_final.py
    python pipeline_final.py --video "C:\\caminho\\16.avi"
    python pipeline_final.py --video x.avi --excel "C:\\...\\Teste MLP.xlsx" --frame_step 5
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Reaproveita o pipeline ja construido
from segment_pipeline import (
    get_yolo_box, get_attention_info, segment_with_sam, annotate_frame,
    DEFAULT_ATTN_TOP1, DEFAULT_YOLO_CONF, OVERSHOOT_RATIO,
)
from train_mlp import WeldMLP, ZScaler
from build_mlp_dataset import COLUMN_KEYWORDS, _normalize, _match_column
from Aplicacao_Enconder import EncoderMAE


HERE = Path(__file__).resolve().parent

DEFAULT_VIDEO       = Path(r"C:\Users\t.soares\Desktop\Treinamento Vision\Inspecao MLP Video\11.avi")
DEFAULT_EXCEL       = Path(r"C:\Users\t.soares\Desktop\Treinamento Vision\Teste MLP.xlsx")
DEFAULT_EXCEL_DIR   = Path(r"C:\Users\t.soares\Desktop\Treinamento Vision\Planilhas MLP Inspecao")
DEFAULT_MLP_CKPT    = HERE / "ckpt_mlp" / "mlp_best.pt"
DEFAULT_OUT_ROOT    = HERE / "Resultados Finais"

DEFAULT_FRAME_STEP  = 5         # processa 1 a cada 5 frames (pedido)


def _find_excel_for_amostra(excel_dir, amostra_id):
    """Busca um Excel correspondente a Amostra na pasta dada.

    Testa varios padroes de nome:
      <id>.xlsx, Amostra_<id>.xlsx, amostra_<id>.xlsx, <id>_*.xlsx, *_<id>.xlsx
    Retorna o primeiro path encontrado, ou None.
    """
    excel_dir = Path(excel_dir)
    if not excel_dir.exists():
        return None
    candidates = [
        excel_dir / f"{amostra_id}.xlsx",
        excel_dir / f"Amostra_{amostra_id}.xlsx",
        excel_dir / f"amostra_{amostra_id}.xlsx",
        excel_dir / f"Amostra {amostra_id}.xlsx",
        excel_dir / f"amostra {amostra_id}.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return c
    # tenta glob por id no nome
    for pattern in (f"*{amostra_id}*.xlsx", f"*{amostra_id}*.xls"):
        for p in excel_dir.glob(pattern):
            return p
    return None

# parametros do processo que vem do Excel (devem casar com o MLP treinado)
PROC_COLS = ["incremento_z", "n_camadas", "altura_ideal",
             "vel_arame", "potencia", "vel_robo"]


# -----------------------------------------------------------------------------
# MLP: carregamento + previsao
# -----------------------------------------------------------------------------
def load_mlp(ckpt_path, device="cpu"):
    """Carrega o MLP treinado + scalers + metadados."""
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = WeldMLP(in_dim=ck["in_dim"], hidden=tuple(ck["hidden"]), drop=ck["drop"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    sx = ZScaler.from_dict(ck["scaler_x"])
    sy = ZScaler.from_dict(ck["scaler_y"])
    return model, sx, sy, list(ck["feat_cols"]), list(ck["target_cols"])


def predict_mlp(model, sx, sy, feat_dict, feat_cols):
    """Roda o MLP com um dict {feature_name: value} e retorna (altura_mm, largura_mm)."""
    try:
        x = np.array([[float(feat_dict[c]) for c in feat_cols]], dtype=np.float64)
    except KeyError as e:
        raise KeyError(f"Feature faltando para o MLP: {e}. Disponivel: {list(feat_dict)}")
    x_z = sx.transform(x).astype(np.float32)
    with torch.no_grad():
        pred_z = model(torch.from_numpy(x_z)).numpy()
    pred = sy.inverse_transform(pred_z)[0]
    return float(pred[0]), float(pred[1])


# -----------------------------------------------------------------------------
# Excel: extrai a linha da Amostra
# -----------------------------------------------------------------------------
def load_test_row(excel_path, amostra_id):
    """Le o Excel de teste e devolve dict {feature_canonica: valor} para a Amostra dada."""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel nao encontrado: {excel_path}")
    df = pd.read_excel(excel_path)

    # mapeia colunas para nomes canonicos via keywords (mesma logica do build_mlp_dataset)
    rename = {}
    for canon, kws in COLUMN_KEYWORDS.items():
        col = _match_column(df.columns, kws)
        if col is not None:
            rename[col] = canon
    df = df.rename(columns=rename)

    # precisa pelo menos da coluna 'amostra' e dos parametros de processo
    needed = ["amostra"] + PROC_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas faltando no Excel: {missing}. Encontradas: {list(df.columns)}")

    df["amostra"] = pd.to_numeric(df["amostra"], errors="coerce")
    row = df.loc[df["amostra"] == amostra_id]
    if len(row) == 0:
        raise ValueError(f"Amostra {amostra_id} nao encontrada no Excel. "
                         f"Disponiveis: {sorted(df['amostra'].dropna().astype(int).tolist())}")
    r = row.iloc[0]
    return {c: float(r[c]) for c in PROC_COLS}


# -----------------------------------------------------------------------------
# Metricas cumulativas
# -----------------------------------------------------------------------------
def compute_running_metrics(areas, A0):
    """Calcula area_mean, area_std, amplitude, frac_overshoot a partir do historico."""
    if len(areas) == 0 or A0 is None or A0 <= 0:
        return {"area_mean": 0.0, "area_std": 0.0, "area_amplitude": 0.0,
                "frac_overshoot": 0.0, "A0": float(A0) if A0 else 0.0}
    arr = np.asarray(areas, dtype=np.float64)
    return {
        "area_mean":      float(arr.mean()),
        "area_std":       float(arr.std()),
        "area_amplitude": float(arr.max() - arr.min()),
        "frac_overshoot": float((arr > OVERSHOOT_RATIO * A0).sum() / len(arr)),
        "A0":             float(A0),
    }


# -----------------------------------------------------------------------------
# Anotacao com overlay de previsao MLP
# -----------------------------------------------------------------------------
def annotate_with_prediction(img_bgr, mask, box, decision, info_text,
                             mlp_pred, ideal=None):
    """Frame anotado: mascara + box + label + caixa com previsao do MLP."""
    out = annotate_frame(img_bgr, mask=mask, box=box, decision=decision,
                         info_text=info_text)
    if mlp_pred is None:
        return out

    altura, largura = mlp_pred
    H_img, W_img = out.shape[:2]

    lines = [
        "PREVISAO MLP",
        f"  altura  = {altura:.2f} mm",
        f"  largura = {largura:.2f} mm",
    ]
    if ideal is not None and "altura_ideal" in ideal:
        lines.append(f"  altura_ideal = {ideal['altura_ideal']:.2f} mm")
        lines.append(f"  diff = {altura - ideal['altura_ideal']:+.2f} mm")

    # caixa preta translucida no canto inferior esquerdo
    line_h = 28
    pad = 8
    block_w = 360
    block_h = line_h * len(lines) + 2 * pad
    x0 = 10
    y0 = H_img - block_h - 10
    cv2.rectangle(out, (x0, y0), (x0 + block_w, y0 + block_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        y = y0 + pad + (i + 1) * line_h - 6
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        thick = 2 if i == 0 else 1
        cv2.putText(out, line, (x0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thick, cv2.LINE_AA)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Pipeline final integrado.")
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--excel", default=None,
                    help="caminho explicito do Excel. Se omitido, busca em --excel_dir "
                         "por arquivo com ID da amostra; fallback p/ Teste MLP.xlsx")
    ap.add_argument("--excel_dir", default=str(DEFAULT_EXCEL_DIR),
                    help="pasta com varias planilhas (1 por amostra). "
                         "Procura <id>.xlsx, Amostra_<id>.xlsx, etc.")
    ap.add_argument("--mlp_ckpt", default=str(DEFAULT_MLP_CKPT))
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--frame_step", type=int, default=DEFAULT_FRAME_STEP)
    ap.add_argument("--attn_threshold", type=float, default=DEFAULT_ATTN_TOP1)
    ap.add_argument("--conf_threshold", type=float, default=DEFAULT_YOLO_CONF)
    ap.add_argument("--amostra", type=int, default=None,
                    help="forca o ID da amostra (default: deduz do stem do video)")
    args = ap.parse_args()

    # limpa eventuais espacos/aspas no path
    video_path = Path(str(args.video).strip().strip('"').strip("'"))
    if not video_path.exists():
        raise FileNotFoundError(f"video nao existe: {video_path}")
    if not Path(args.mlp_ckpt).exists():
        raise FileNotFoundError(f"checkpoint MLP nao existe: {args.mlp_ckpt}")

    vid = video_path.stem
    amostra_id = args.amostra if args.amostra is not None else int(float(vid))

    # === Resolve o caminho do Excel: prioridade CLI > pasta dedicada > default ===
    if args.excel is not None:
        excel_path = Path(str(args.excel).strip().strip('"').strip("'"))
        print(f"[excel] usando caminho explicito: {excel_path}")
    else:
        found = _find_excel_for_amostra(args.excel_dir, amostra_id)
        if found is not None:
            excel_path = found
            print(f"[excel] encontrado por ID na pasta: {excel_path}")
        else:
            excel_path = Path(str(DEFAULT_EXCEL).strip())
            print(f"[excel] fallback p/ default: {excel_path}")
    if not excel_path.exists():
        raise FileNotFoundError(f"excel nao existe: {excel_path}")

    # === Carrega modelos ===
    print("[init] carregando YOLO + Vision Encoder + SAM + MLP...")
    encoder = EncoderMAE()
    yolo = encoder.yolo

    from ultralytics import SAM
    sam_pt = HERE / "sam_b.pt"
    if not sam_pt.exists():
        print("[init] SAM weights ausente; ultralytics vai baixar sam_b.pt")
        sam_pt = "sam_b.pt"
    sam = SAM(str(sam_pt))

    mlp, sx, sy, feat_cols, target_cols = load_mlp(args.mlp_ckpt)
    print(f"[init] MLP: in_dim={len(feat_cols)} feats={feat_cols} | targets={target_cols}")

    # === Le Excel ===
    print(f"[excel] amostra {amostra_id}: {excel_path.name}")
    proc_params = load_test_row(excel_path, amostra_id)
    print(f"[excel] parametros do processo: {proc_params}")

    # === Prepara saidas ===
    out_dir = Path(args.out_root) / vid
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) inputs
    pd.DataFrame([{"amostra": amostra_id, **proc_params}]).to_csv(
        out_dir / f"{vid}_inputs.csv", index=False
    )

    # 2) abre video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"nao foi possivel abrir video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = out_dir / f"{vid}_reconstructed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"nao foi possivel criar video saida: {out_video_path}")

    # 3) CSVs em streaming
    metrics_f = open(out_dir / f"{vid}_metrics_realtime.csv", "w", newline="", encoding="utf-8")
    metrics_w = csv.writer(metrics_f)
    metrics_w.writerow(["frame_idx", "decision", "max_top1", "area_px",
                        "A0", "area_mean", "area_std", "area_amplitude", "frac_overshoot"])

    pred_f = open(out_dir / f"{vid}_mlp_predictions.csv", "w", newline="", encoding="utf-8")
    pred_w = csv.writer(pred_f)
    pred_w.writerow(["frame_idx", "altura_pred_mm", "largura_pred_mm"])

    # === Loop ===
    A0 = None
    areas = []
    n_seg = 0
    last_state = {
        "mask": None, "box": None, "decision": "INIT", "info_text": "",
        "mlp_pred": None,
    }
    last_pred = None

    pbar = tqdm(total=n_total if n_total > 0 else None, desc=f"video {vid}")
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            do_proc = (idx % args.frame_step == 0)
            if do_proc:
                # YOLO
                yolo_box = get_yolo_box(yolo, frame, args.conf_threshold, target_class=None)
                # atencao do Vision
                attn_info = get_attention_info(encoder, frame)
                has_yolo = yolo_box is not None
                has_attn = attn_info["max_top1"] > args.attn_threshold

                decision = "?"
                mask = None
                if has_yolo and has_attn:
                    decision = "BOX+ATTN"
                    mask = segment_with_sam(sam, frame, box=yolo_box)
                elif has_attn and not has_yolo:
                    decision = "SKIP_NO_BOX"
                elif has_yolo and not has_attn:
                    decision = "SKIP_NO_ATTN"
                else:
                    decision = "SKIP_NO_BOTH"

                area = int(mask.sum()) if mask is not None else 0
                if area > 0:
                    if A0 is None:
                        A0 = area
                    areas.append(area)
                    n_seg += 1

                # metricas cumulativas
                rm = compute_running_metrics(areas, A0)

                # MLP prediction (so se a metrica esta minimamente consolidada)
                if A0 is not None and n_seg >= 1:
                    feat_dict = {**proc_params, **rm}
                    altura, largura = predict_mlp(mlp, sx, sy, feat_dict, feat_cols)
                    last_pred = (altura, largura)
                    pred_w.writerow([idx, f"{altura:.4f}", f"{largura:.4f}"])

                # log de metricas
                metrics_w.writerow([
                    idx, decision, f"{attn_info['max_top1']:.4f}", area,
                    f"{rm['A0']:.2f}", f"{rm['area_mean']:.2f}",
                    f"{rm['area_std']:.2f}", f"{rm['area_amplitude']:.2f}",
                    f"{rm['frac_overshoot']:.4f}",
                ])

                # atualiza estado da overlay
                ratio = (area / A0) if (A0 and area > 0) else 0.0
                last_state = {
                    "mask":      mask,
                    "box":       yolo_box,
                    "decision":  decision,
                    "info_text": f"top1={attn_info['max_top1']:.3f} area={area} ratio={ratio:.2f}",
                    "mlp_pred":  last_pred,
                }

            # anota frame (usa o ultimo estado processado)
            annotated = annotate_with_prediction(
                frame,
                mask=last_state["mask"],
                box=last_state["box"],
                decision=last_state["decision"],
                info_text=last_state["info_text"],
                mlp_pred=last_state["mlp_pred"],
                ideal={"altura_ideal": proc_params["altura_ideal"]},
            )
            writer.write(annotated)

            idx += 1
            pbar.update(1)
    finally:
        cap.release()
        writer.release()
        metrics_f.close()
        pred_f.close()
        pbar.close()

    # === resumo final ===
    final_summary = {
        "video":          str(video_path),
        "amostra_id":     amostra_id,
        "n_frames_total": idx,
        "n_processed":    (idx + args.frame_step - 1) // args.frame_step,
        "n_segmented":    n_seg,
        "frame_step":     args.frame_step,
        "A0":             float(A0) if A0 else 0.0,
        "proc_params":    proc_params,
        "final_metrics":  compute_running_metrics(areas, A0),
        "final_prediction": {
            "altura_pred_mm":  last_pred[0] if last_pred else None,
            "largura_pred_mm": last_pred[1] if last_pred else None,
        },
    }
    with open(out_dir / f"{vid}_final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {idx} frames | {n_seg} segmentados (1/{args.frame_step})")
    print(f"     A0 baseline      : {A0}")
    if last_pred:
        print(f"     altura prevista  : {last_pred[0]:.2f} mm")
        print(f"     largura prevista : {last_pred[1]:.2f} mm")
    print(f"     pasta de saida   : {out_dir}")
    print(f"       inputs.csv     : {out_dir / (vid + '_inputs.csv')}")
    print(f"       metrics_realtime.csv: {out_dir / (vid + '_metrics_realtime.csv')}")
    print(f"       mlp_predictions.csv : {out_dir / (vid + '_mlp_predictions.csv')}")
    print(f"       final_summary.json  : {out_dir / (vid + '_final_summary.json')}")
    print(f"       reconstructed.mp4   : {out_video_path}")


if __name__ == "__main__":
    main()
