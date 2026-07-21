"""
Aplicacao do Encoder MAE treinado.

Este script aplica o pipeline completo (YOLO frozen -> ExternalFPN ->
PatchEmbedding -> blocos do encoder MAE) sobre uma imagem qualquer e
retorna a feature que voce vai usar como input do MLP.

NAO usa decoder, mask_token nem pred -> esses so serviram no pretraining.

Modos suportados (parametro feature_type):
    "attn_mean"  -> media das heads do mapa de atencao do ULTIMO bloco
                    shape [B, N*N]   ex: [1, 625]   <- recomendado p/ MLP simples
    "attn_full"  -> mapa completo por head do ULTIMO bloco
                    shape [B, H*N*N] ex: [1, 2500]
    "embed_mean" -> media dos tokens do encoder (pooled embedding)
                    shape [B, D]     ex: [1, 384]
    "embed_flat" -> todos os tokens concatenados
                    shape [B, N*D]   ex: [1, 9600]
    "tokens"     -> tokens crus (nao achatado), uteis para attention rollout
                    shape [B, N, D]  ex: [1, 25, 384]

Uso programatico:
    from Aplicação_Enconder import EncoderMAE
    enc = EncoderMAE(yolo_pt="best.pt", mae_pt="ckpt_mae/mae_epoch_020.pt")
    feat = enc.encode_image("alguma_imagem.jpg", feature_type="attn_mean")
    print(feat.shape)   # torch.Size([1, 625])

Uso pela linha de comando (teste rapido):
    python "Aplicação_Enconder" --img caminho/para/foto.jpg --feature attn_mean
"""

import argparse
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reaproveita as classes do treino MAE (mesmo diretorio)
from train_mae_local import (
    ExternalFPN, PatchEmbedding, TransformerBlock,
    IDX_P3, IDX_P4, IDX_P5, IMG_SIZE,
)


HERE = Path(__file__).resolve().parent
DEFAULT_YOLO_PT = Path(r"C:\Users\t.soares\Desktop\Treinamento Vision\best.pt")
DEFAULT_MAE_PT  = Path(r"C:\Users\t.soares\Desktop\Treinamento Vision\ckpt_mae_attn\mae_epoch_020.pt")


class EncoderMAE:
    """Wrapper que carrega YOLO + Encoder MAE e produz a feature de uma imagem."""

    def __init__(self, yolo_pt=DEFAULT_YOLO_PT, mae_pt=DEFAULT_MAE_PT, device=None):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.yolo_pt = Path(yolo_pt)
        self.mae_pt = Path(mae_pt)

        if not self.yolo_pt.exists():
            raise FileNotFoundError(f"YOLO weights nao encontrado: {self.yolo_pt}")
        if not self.mae_pt.exists():
            raise FileNotFoundError(f"Checkpoint MAE nao encontrado: {self.mae_pt}")

        self._load_yolo()
        self._load_mae()

    # -----------------------------------------------------------------
    def _load_yolo(self):
        from ultralytics import YOLO
        self.yolo = YOLO(str(self.yolo_pt))   # mantem objeto YOLO p/ predict()
        self.net = self.yolo.model.eval().to(self.device)
        for p in self.net.parameters():
            p.requires_grad = False

        self._feats = {}
        def hook(name):
            def _h(_m, _i, out):
                self._feats[name] = out
            return _h

        self._h3 = self.net.model[IDX_P3].register_forward_hook(hook("P3"))
        self._h4 = self.net.model[IDX_P4].register_forward_hook(hook("P4"))
        self._h5 = self.net.model[IDX_P5].register_forward_hook(hook("P5"))

    # -----------------------------------------------------------------
    def _load_mae(self):
        ck = torch.load(self.mae_pt, map_location=self.device, weights_only=False)
        cfg = ck["config"]
        sd = ck["state_dict"]
        self.cfg = cfg

        self.fpn = ExternalFPN(cfg["c3"], cfg["c4"], cfg["c5"], outc=64).eval().to(self.device)
        self.patch = PatchEmbedding(64 * 3, cfg["EMBED_DIM"], cfg["PATCH_SIZE"]).eval().to(self.device)
        self.encoder = nn.ModuleList([
            TransformerBlock(cfg["EMBED_DIM"], cfg["NUM_HEADS"], cfg["MLP_RATIO"], cfg["DROPOUT"])
            .eval().to(self.device)
            for _ in range(cfg["ENC_DEPTH"])
        ])

        # carrega so as chaves relevantes (FPN, PatchEmbed, pos_enc, blocos do encoder)
        self.fpn.load_state_dict({k[len("fpn."):]: v for k, v in sd.items() if k.startswith("fpn.")})
        self.patch.load_state_dict({k[len("patch."):]: v for k, v in sd.items() if k.startswith("patch.")})
        self.pos_enc = sd["pos_enc"].to(self.device)
        for i, blk in enumerate(self.encoder):
            pref = f"encoder.{i}."
            blk.load_state_dict({k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)})

        for p in list(self.fpn.parameters()) + list(self.patch.parameters()) + \
                 [p for blk in self.encoder for p in blk.parameters()]:
            p.requires_grad = False

        self.fmap_size = cfg["fmap_size"]
        print(f"[EncoderMAE] carregado: epoch={ck['epoch']} avg_loss={ck['avg_loss']:.4f} | "
              f"EMBED_DIM={cfg['EMBED_DIM']} HEADS={cfg['NUM_HEADS']} ENC_DEPTH={cfg['ENC_DEPTH']}")

    # -----------------------------------------------------------------
    def _read_and_resize(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return x.to(self.device)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, x):
        """x: [B, 3, IMG_SIZE, IMG_SIZE] -> (tokens [B,N,D], attn_last [B,H,N,N])."""
        self._feats.clear()
        _ = self.net(x)
        P3 = self._feats["P3"]
        P4 = self._feats["P4"]
        P5 = self._feats["P5"]

        f = self.fpn(P3, P4, P5)
        f = F.interpolate(f, size=(self.fmap_size, self.fmap_size),
                          mode="bilinear", align_corners=False)
        f = f.float()
        f = (f - f.mean(dim=[2, 3], keepdim=True)) / (f.std(dim=[2, 3], keepdim=True) + 1e-6)

        z = self.patch(f) + self.pos_enc

        last_idx = len(self.encoder) - 1
        attn_last = None
        for i, blk in enumerate(self.encoder):
            z2 = blk.ln1(z)
            a, w = blk.attn(z2, z2, z2, need_weights=True, average_attn_weights=False)
            if i == last_idx:
                attn_last = w  # [B, H, N, N]
            z = z + a
            z = z + blk.ffn(blk.ln2(z))
        return z, attn_last

    # -----------------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, img_path, feature_type="attn_mean"):
        """Aplica o pipeline completo numa imagem e retorna a feature."""
        x = self._read_and_resize(img_path)
        tokens, attn_last = self._forward(x)
        return self._select_feature(tokens, attn_last, feature_type)

    @torch.no_grad()
    def encode_batch(self, img_paths, feature_type="attn_mean"):
        """Mesma coisa, em lote (lista de caminhos)."""
        xs = [self._read_and_resize(p) for p in img_paths]
        x = torch.cat(xs, dim=0)
        tokens, attn_last = self._forward(x)
        return self._select_feature(tokens, attn_last, feature_type)

    # -----------------------------------------------------------------
    @staticmethod
    def _select_feature(tokens, attn_last, feature_type):
        # tokens: [B, N, D] | attn_last: [B, H, N, N]
        if feature_type == "attn_mean":
            return attn_last.mean(dim=1).flatten(1)         # [B, N*N]
        if feature_type == "attn_full":
            return attn_last.flatten(1)                     # [B, H*N*N]
        if feature_type == "embed_mean":
            return tokens.mean(dim=1)                       # [B, D]
        if feature_type == "embed_flat":
            return tokens.flatten(1)                        # [B, N*D]
        if feature_type == "tokens":
            return tokens                                   # [B, N, D]
        raise ValueError(f"feature_type invalido: {feature_type}")

    # -----------------------------------------------------------------
    def __del__(self):
        # tira os hooks ao destruir o objeto
        for h in (getattr(self, "_h3", None), getattr(self, "_h4", None), getattr(self, "_h5", None)):
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass


# ===================================================================
# CLI de teste rapido
# ===================================================================
def main():
    ap = argparse.ArgumentParser(description="Aplica encoder MAE treinado em uma imagem.")
    ap.add_argument("--img", required=True, help="caminho para a imagem")
    ap.add_argument("--yolo", default=str(DEFAULT_YOLO_PT), help="path do .pt do YOLO")
    ap.add_argument("--mae", default=str(DEFAULT_MAE_PT), help="path do .pt do MAE")
    ap.add_argument("--feature", default="attn_mean",
                    choices=["attn_mean", "attn_full", "embed_mean", "embed_flat", "tokens"])
    ap.add_argument("--save", default=None, help="se passado, salva a feature em .pt")
    args = ap.parse_args()

    enc = EncoderMAE(yolo_pt=args.yolo, mae_pt=args.mae)
    feat = enc.encode_image(args.img, feature_type=args.feature)

    print(f"[OK] imagem: {args.img}")
    print(f"     feature_type: {args.feature}")
    print(f"     shape: {tuple(feat.shape)}")
    print(f"     dtype: {feat.dtype}")
    print(f"     min/max: {feat.min().item():.4f} / {feat.max().item():.4f}")
    print(f"     mean/std: {feat.mean().item():.4f} / {feat.std().item():.4f}")

    if args.save:
        torch.save({"feature": feat.cpu(), "type": args.feature, "src": args.img}, args.save)
        print(f"     salvo em: {args.save}")


if __name__ == "__main__":
    main()
