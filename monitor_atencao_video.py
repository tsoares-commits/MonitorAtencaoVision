import argparse
import json
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import jensenshannon
import torch

from inspecionar_atencao import _import_encoder_module, _heatmap_overlay, _entropy, _put_label

IMG_SIZE = 640

def calculate_jsd_inter_heads(attn):
    """ Calcula a divergencia de Jensen-Shannon media entre as heads """
    H, N, _ = attn.shape
    marginals = [attn[h].mean(axis=0) for h in range(H)]
    mean_dist = np.mean(marginals, axis=0)
    jsds = [jensenshannon(marg, mean_dist) for marg in marginals]
    # JS divergence retornado by scipy is the distance, so we square it to get divergence, 
    # but distance is also fine. We'll use distance (sqrt of div) as it's linear.
    return np.mean(jsds)

def get_attention_metrics_for_frame(enc, img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(enc.device) / 255.0

    with torch.no_grad():
        _tokens, attn_last = enc._forward(x)
        
    attn = attn_last.squeeze(0).cpu().numpy() # [H, N, N]
    H, N, _ = attn.shape
    grid = int(round(np.sqrt(N)))
    h_step = IMG_SIZE // grid
    w_step = IMG_SIZE // grid

    metrics = {}
    centroids = []
    
    # Por head
    for h in range(H):
        marginal = attn[h].mean(axis=0)
        e = _entropy(marginal)
        e_norm = e / np.log(N) if N > 1 else 0
        top1 = float(marginal.max())
        
        # Centroid
        marginal_2d = marginal.reshape(grid, grid)
        # malha de coordenadas dos centros dos patches
        xs = np.linspace(w_step/2, IMG_SIZE - w_step/2, grid)
        ys = np.linspace(h_step/2, IMG_SIZE - h_step/2, grid)
        xv, yv = np.meshgrid(xs, ys)
        
        cx = np.sum(xv * marginal_2d)
        cy = np.sum(yv * marginal_2d)
        centroids.append(np.array([cx, cy]))
        
        metrics[f'entropy_norm_h{h+1}'] = e_norm
        metrics[f'top1_share_h{h+1}'] = top1
        metrics[f'centroid_x_h{h+1}'] = cx
        metrics[f'centroid_y_h{h+1}'] = cy

    # Agregadas
    metrics['entropy_mean'] = np.mean([metrics[f'entropy_norm_h{h+1}'] for h in range(H)])
    metrics['top1_std'] = np.std([metrics[f'top1_share_h{h+1}'] for h in range(H)])
    metrics['jsd_inter_heads'] = calculate_jsd_inter_heads(attn)
    
    return metrics, attn, centroids

def render_panel(img_bgr, attn, centroids, metrics, frame_idx, out_path, prev_centroids=None):
    H = attn.shape[0]
    grid = int(round(np.sqrt(attn.shape[1])))
    
    # Resize original pra 320x320 pra ficar menor no painel
    small_size = 320
    orig_small = cv2.resize(img_bgr, (small_size, small_size))
    
    panels = [_put_label(orig_small.copy(), f"Frame {frame_idx}")]
    
    for h in range(H):
        marginal_2d = attn[h].mean(axis=0).reshape(grid, grid)
        over = _heatmap_overlay(img_bgr, marginal_2d, alpha=0.5)
        
        # Desenha centroide
        cx, cy = int(metrics[f'centroid_x_h{h+1}']), int(metrics[f'centroid_y_h{h+1}'])
        cv2.circle(over, (cx, cy), 8, (255,255,255), -1)
        cv2.circle(over, (cx, cy), 8, (0,0,0), 2)
        
        if prev_centroids is not None:
            px, py = int(prev_centroids[h][0]), int(prev_centroids[h][1])
            cv2.arrowedLine(over, (px, py), (cx, cy), (0, 255, 255), 3, tipLength=0.3)
            delta = np.linalg.norm(np.array([cx, cy]) - np.array([px, py]))
        else:
            delta = 0

        # Labels
        en = metrics[f'entropy_norm_h{h+1}']
        top1 = metrics[f'top1_share_h{h+1}']
        over = _put_label(over, f"H{h+1} | E={en:.2f} T={top1:.2f} D={delta:.0f}px")
        
        over_small = cv2.resize(over, (small_size, small_size))
        panels.append(over_small)

    # Média
    mean_2d = attn.mean(axis=0).mean(axis=0).reshape(grid, grid)
    over_mean = _heatmap_overlay(img_bgr, mean_2d, alpha=0.5)
    e_mean = metrics['entropy_mean']
    jsd = metrics['jsd_inter_heads']
    
    if e_mean < 0.4 and jsd < 0.05:
        estado = "ESTAVEL"
        cor = (0, 255, 0)
    elif e_mean < 0.6 and jsd < 0.15:
        estado = "TRANSICAO"
        cor = (0, 255, 255)
    else:
        estado = "INSTAVEL"
        cor = (0, 0, 255)
        
    cv2.putText(over_mean, f"Estado: {estado}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor, 3)
    over_mean = _put_label(over_mean, f"Media | E={e_mean:.2f} JSD={jsd:.3f}")
    
    panels.append(cv2.resize(over_mean, (small_size, small_size)))
    
    # Concatena lado a lado
    final_panel = np.hstack(panels)
    cv2.imwrite(str(out_path), final_panel)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Caminho do video')
    parser.add_argument('--label', required=True, choices=['estavel', 'instavel'])
    parser.add_argument('--frame_step', type=int, default=5)
    parser.add_argument('--mae', default='ckpt_mae_attn/mae_epoch_020.pt')
    parser.add_argument('--out_dir', default='out_monitor')
    args = parser.parse_args()

    EncoderMAE = _import_encoder_module()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Instancia encoder
    enc = EncoderMAE(
        img_size=IMG_SIZE, patch_size=128, in_chans=3,
        embed_dim=256, depth=2, num_heads=4,
        decoder_embed_dim=128, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=torch.nn.LayerNorm
    ).to(device)
    
    print(f"Carregando pesos: {args.mae}")
    checkpoint = torch.load(args.mae, map_location=device)
    if 'model' in checkpoint:
        enc.load_state_dict(checkpoint['model'], strict=False)
    else:
        enc.load_state_dict(checkpoint, strict=False)
    enc.eval()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Erro ao abrir {args.video}")
        return

    video_name = Path(args.video).stem
    out_video_dir = Path(args.out_dir) / video_name
    heat_dir = out_video_dir / "heatmaps"
    plot_dir = out_video_dir / "plots"
    heat_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    records = []
    prev_centroids = None
    
    print("Processando video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % args.frame_step == 0:
            metrics, attn, centroids = get_attention_metrics_for_frame(enc, frame)
            
            # Calcula deslocamento
            if prev_centroids is not None:
                disps = [np.linalg.norm(centroids[h] - prev_centroids[h]) for h in range(4)]
                metrics['centroid_disp_mean'] = np.mean(disps)
                for h in range(4):
                    metrics[f'centroid_disp_h{h+1}'] = disps[h]
            else:
                metrics['centroid_disp_mean'] = 0.0
                for h in range(4):
                    metrics[f'centroid_disp_h{h+1}'] = 0.0
            
            metrics['frame'] = frame_idx
            records.append(metrics)
            
            # Render painel
            out_panel = heat_dir / f"frame_{frame_idx:05d}.png"
            render_panel(frame, attn, centroids, metrics, frame_idx, out_panel, prev_centroids)
            
            prev_centroids = centroids
            
        frame_idx += 1

    cap.release()
    
    # Processa dataset e calcula TEV (Temporal Entropy Variance)
    df = pd.DataFrame(records)
    window = 15 // args.frame_step
    if window < 2: window = 2
    
    df['tev'] = df['entropy_mean'].rolling(window=window, min_periods=1).var().fillna(0)
    df.to_csv(out_video_dir / "metrics_por_frame.csv", index=False)
    
    # Gera plots
    plt.figure(figsize=(10, 4))
    for h in range(4):
        plt.plot(df['frame'], df[f'entropy_norm_h{h+1}'], label=f'H{h+1}', alpha=0.6)
    plt.plot(df['frame'], df['entropy_mean'], 'k--', label='Media', linewidth=2)
    plt.title(f"Entropia Espacial - {video_name}")
    plt.legend()
    plt.savefig(plot_dir / "entropy_series.png")
    plt.close()
    
    plt.figure(figsize=(10, 4))
    plt.plot(df['frame'], df['jsd_inter_heads'], 'r-', linewidth=2)
    plt.axhline(0.05, color='g', linestyle='--')
    plt.axhline(0.15, color='r', linestyle='--')
    plt.title(f"JSD Inter-Heads - {video_name}")
    plt.savefig(plot_dir / "jsd_series.png")
    plt.close()
    
    plt.figure(figsize=(10, 4))
    plt.plot(df['frame'], df['tev'], 'b-', linewidth=2)
    plt.title(f"TEV (Temporal Entropy Variance) - {video_name}")
    plt.savefig(plot_dir / "tev_series.png")
    plt.close()
    
    # Summary JSON e Tabela Resumo
    cond_estavel = (df['entropy_mean'] < 0.4) & (df['jsd_inter_heads'] < 0.05)
    cond_instavel = (df['entropy_mean'] >= 0.6) | (df['jsd_inter_heads'] >= 0.15)
    cond_trans = ~(cond_estavel | cond_instavel)
    
    total = len(df)
    resumo = {
        'Video': video_name,
        'Label': args.label,
        'H_mean': df['entropy_mean'].mean(),
        'H_std': df['entropy_mean'].std(),
        'JSD_mean': df['jsd_inter_heads'].mean(),
        'JSD_std': df['jsd_inter_heads'].std(),
        'PDV_mean': df['centroid_disp_mean'].mean(),
        'TEV_mean': df['tev'].mean(),
        '% Estavel': (cond_estavel.sum() / total) * 100,
        '% Transicao': (cond_trans.sum() / total) * 100,
        '% Instavel': (cond_instavel.sum() / total) * 100
    }
    
    pd.DataFrame([resumo]).to_csv(out_video_dir / "tabela_resumo.csv", index=False)
    with open(out_video_dir / "summary.json", 'w') as f:
        json.dump(resumo, f, indent=4)
        
    print(f"Finalizado {video_name}. Salvo em {out_video_dir}")

if __name__ == '__main__':
    main()
