"""
build_mlp_dataset.py
--------------------
Monta o dataset final para o MLP a partir de:
    1) Parametros do processo (Excel: Parametros de Ensaios.xlsx)
    2) Metricas agregadas de cada video segmentado (<id>_metrics.csv em
       segmentation_output/)

Casamento: coluna 'Amostra' do Excel == video_id do CSV (stem do video).
Ex: video '4.avi' -> Amostra == 4

Saida: mlp_dataset/features.csv com 1 linha por amostra contendo:
    [process params]  +  [area metrics]  +  [targets: altura, largura]

IMPORTANTE: feche o Excel antes de rodar (evita bloqueio do arquivo).

Uso:
    python build_mlp_dataset.py
    python build_mlp_dataset.py --excel "outro\\caminho.xlsx" --metrics_dir saida
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_EXCEL       = HERE / "Parametros de Ensaios.xlsx"
DEFAULT_METRICS_DIR = HERE / "segmentation_output"
DEFAULT_OUT         = HERE / "mlp_dataset" / "features.csv"

# Colunas do Excel -> nome canonical (sem acentos/espacos)
# usa matching por palavras-chave (case-insensitive) p/ ser robusto a variacoes
COLUMN_KEYWORDS = {
    "amostra":       ["amostra"],
    "altura_total":  ["altura", "total"],
    "largura":       ["largura"],
    "incremento_z":  ["incremento"],
    "n_camadas":     ["camadas"],
    "altura_ideal":  ["altura", "ideal"],
    "diff_altura":   ["diferenca", "altura"],
    "vel_arame":     ["velocidade", "alimentacao"],
    "potencia":      ["potencia"],
    "vel_robo":      ["velocidade", "robo"],
}

# Features de processo que entram no MLP (input)
PROC_COLS = ["incremento_z", "n_camadas", "altura_ideal",
             "vel_arame", "potencia", "vel_robo"]

# Saidas que o MLP deve prever (target)
TARGET_COLS = ["altura_total", "largura"]

# Features derivadas do video (input)
AREA_COLS = ["area_mean", "area_std", "area_amplitude", "frac_overshoot", "A0"]


# -------------------------------------------------------------------
# Util: normaliza string para matching robusto a acentos
# -------------------------------------------------------------------
def _normalize(s):
    s = str(s).lower()
    repl = {"á": "a", "à": "a", "â": "a", "ã": "a",
            "é": "e", "ê": "e",
            "í": "i",
            "ó": "o", "ô": "o", "õ": "o",
            "ú": "u", "ü": "u",
            "ç": "c", "°": "", "º": "", "ª": ""}
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _match_column(df_columns, keywords):
    """Acha a primeira coluna do dataframe que contem TODAS as keywords (apos normalizar)."""
    kws = [_normalize(k) for k in keywords]
    for c in df_columns:
        cn = _normalize(c)
        if all(k in cn for k in kws):
            return c
    return None


# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", default=str(DEFAULT_EXCEL),
                    help="caminho do .xlsx com parametros dos ensaios")
    ap.add_argument("--metrics_dir", default=str(DEFAULT_METRICS_DIR),
                    help="pasta com os <id>_metrics.csv do segment_pipeline.py")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="caminho do features.csv de saida")
    args = ap.parse_args()

    excel_path = Path(args.excel)
    if not excel_path.exists():
        # tenta tirar o prefixo ~$ caso seja arquivo de bloqueio
        alt = excel_path.parent / excel_path.name.replace("~$", "")
        if alt.exists():
            excel_path = alt
            print(f"[info] usando arquivo real '{alt.name}' (sem prefixo ~$)")
        else:
            raise FileNotFoundError(f"Excel nao encontrado: {args.excel}")

    # ===== 1) carrega Excel =====
    print(f"[excel] lendo: {excel_path}")
    try:
        df_ens = pd.read_excel(excel_path)
    except Exception as e:
        print(f"[erro] nao consegui abrir o Excel. Talvez esteja aberto no Office? Erro: {e}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[excel] {len(df_ens)} linhas | colunas originais: {list(df_ens.columns)}")

    # mapeia colunas via keywords
    rename_map = {}
    for canon, kws in COLUMN_KEYWORDS.items():
        col = _match_column(df_ens.columns, kws)
        if col is not None:
            rename_map[col] = canon
    df_ens = df_ens.rename(columns=rename_map)

    missing = [c for c in (["amostra"] + PROC_COLS + TARGET_COLS) if c not in df_ens.columns]
    if missing:
        print(f"[erro] colunas nao encontradas no Excel: {missing}", file=sys.stderr)
        print(f"       colunas mapeadas: {list(df_ens.columns)}", file=sys.stderr)
        sys.exit(1)

    needed = ["amostra"] + PROC_COLS + TARGET_COLS
    df_ens = df_ens[needed].dropna(subset=["amostra"] + TARGET_COLS).copy()
    df_ens["amostra"] = pd.to_numeric(df_ens["amostra"], errors="coerce")
    df_ens = df_ens.dropna(subset=["amostra"])
    df_ens["amostra"] = df_ens["amostra"].astype(int)
    print(f"[excel] {len(df_ens)} amostras validas apos limpeza")

    # ===== 2) carrega *_metrics.csv =====
    metrics_dir = Path(args.metrics_dir)
    if not metrics_dir.exists():
        print(f"[erro] pasta de metricas nao existe: {metrics_dir}", file=sys.stderr)
        sys.exit(1)
    metrics_files = sorted(glob.glob(str(metrics_dir / "*_metrics.csv")))
    if not metrics_files:
        print(f"[erro] nenhum *_metrics.csv em {metrics_dir}", file=sys.stderr)
        print( "       rode antes: python segment_pipeline.py --video <arquivo>", file=sys.stderr)
        sys.exit(1)
    print(f"[metrics] {len(metrics_files)} arquivos em {metrics_dir}")

    df_areas_list = []
    for fp in metrics_files:
        df = pd.read_csv(fp)
        if "video_id" not in df.columns:
            print(f"[skip] {Path(fp).name}: sem coluna 'video_id'")
            continue
        df_areas_list.append(df)
    if not df_areas_list:
        print("[erro] nenhum CSV teve formato valido", file=sys.stderr)
        sys.exit(1)
    df_areas = pd.concat(df_areas_list, ignore_index=True)

    df_areas["amostra"] = pd.to_numeric(df_areas["video_id"], errors="coerce")
    df_areas = df_areas.dropna(subset=["amostra"]).copy()
    df_areas["amostra"] = df_areas["amostra"].astype(int)
    # mantem so as colunas que nos interessam
    area_keep = ["amostra"] + [c for c in AREA_COLS if c in df_areas.columns]
    missing_area = [c for c in AREA_COLS if c not in df_areas.columns]
    if missing_area:
        print(f"[warn] colunas de area faltando nos CSVs: {missing_area}")
    df_areas = df_areas[area_keep]
    print(f"[metrics] {len(df_areas)} amostras com metricas")

    # ===== 3) merge por amostra =====
    df = pd.merge(df_ens, df_areas, on="amostra", how="inner")
    print(f"[merged] {len(df)} amostras com Excel + metricas")
    set_ex  = set(df_ens["amostra"])
    set_me  = set(df_areas["amostra"])
    only_ex = sorted(set_ex - set_me)
    only_me = sorted(set_me - set_ex)
    if only_ex:
        print(f"[warn] amostras no Excel SEM video segmentado: {only_ex}")
    if only_me:
        print(f"[warn] videos segmentados SEM linha no Excel: {only_me}")

    if len(df) < 5:
        print("[erro] menos de 5 amostras alinhadas — nao da pra treinar MLP", file=sys.stderr)
        sys.exit(1)

    # ===== 4) salva features.csv =====
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[OK] features salvas em: {out_path}")
    print(f"     amostras  : {len(df)}")
    print(f"     colunas   : {list(df.columns)}")
    print(f"     n_features: {len(PROC_COLS) + len(AREA_COLS)}")
    print(f"     n_targets : {len(TARGET_COLS)}")

    # ===== 5) sumario =====
    print("\n[stats] resumo das features (input do MLP):")
    feat_cols = PROC_COLS + [c for c in AREA_COLS if c in df.columns]
    print(df[feat_cols].describe().T[["mean", "std", "min", "max"]].round(3))
    print("\n[stats] resumo dos targets:")
    print(df[TARGET_COLS].describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
