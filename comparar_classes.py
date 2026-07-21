import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='out_monitor', help='Pasta com as saidas dos videos')
    args = parser.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        print(f"Pasta {args.base_dir} nao encontrada.")
        return

    all_summaries = []
    
    # Procura todos os tabela_resumo.csv nas subpastas
    for csv_file in base.rglob('tabela_resumo.csv'):
        df = pd.read_csv(csv_file)
        all_summaries.append(df)
        
    if not all_summaries:
        print("Nenhum CSV de resumo encontrado. Rode o monitor_atencao_video.py primeiro.")
        return
        
    df_all = pd.concat(all_summaries, ignore_index=True)
    
    # Salva tabela agregada
    df_all.to_csv(base / "tabela_geral_artigo.csv", index=False)
    print("\nTabela geral por video:")
    print(df_all[['Video', 'Label', 'H_mean', 'JSD_mean', '% Estavel', '% Instavel']])
    
    # Comparacao estatistica entre Estavel e Instavel
    estavel = df_all[df_all['Label'] == 'estavel']
    instavel = df_all[df_all['Label'] == 'instavel']
    
    metrics_to_compare = ['H_mean', 'JSD_mean', 'PDV_mean', 'TEV_mean']
    
    results = []
    
    for m in metrics_to_compare:
        val_est = estavel[m].values
        val_inst = instavel[m].values
        
        if len(val_est) > 0 and len(val_inst) > 0:
            # Mann-Whitney U test (nao parametrico)
            stat, pval = stats.mannwhitneyu(val_est, val_inst, alternative='two-sided')
            
            mean_est, std_est = np.mean(val_est), np.std(val_est)
            mean_inst, std_inst = np.mean(val_inst), np.std(val_inst)
            
            cd = cohen_d(val_inst, val_est) # effect size de instavel vs estavel
            
            results.append({
                'Metrica': m,
                'Estavel (mu+-std)': f"{mean_est:.3f}+-{std_est:.3f}",
                'Instavel (mu+-std)': f"{mean_inst:.3f}+-{std_inst:.3f}",
                'p-valor': pval,
                'Cohen d': cd
            })
            
    df_comp = pd.DataFrame(results)
    df_comp.to_csv(base / "comparativo_classes_artigo.csv", index=False)
    print("\nComparativo Estatistico:")
    print(df_comp)
    
    # Boxplots
    if len(estavel) > 0 and len(instavel) > 0:
        plt.figure(figsize=(12, 10))
        for i, m in enumerate(metrics_to_compare):
            plt.subplot(2, 2, i+1)
            sns.boxplot(x='Label', y=m, data=df_all)
            plt.title(m)
        plt.tight_layout()
        plt.savefig(base / "boxplots_classes.png")
        plt.close()
        print(f"\nBoxplots salvos em {base / 'boxplots_classes.png'}")

if __name__ == '__main__':
    main()
