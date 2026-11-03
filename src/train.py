"""
Fine-tuning do YOLO (pré-treinado no COCO, detector genérico) para detectar e classificar
letras do alfabeto manual de Libras.

Estratégia:
- Parte de pesos COCO (transfer learning): o backbone já sabe extrair bordas/formas,
  a cabeça de detecção é reinicializada para as 15 letras e toda a rede é ajustada
  (`--freeze N` permite congelar as N primeiras camadas para comparação).
- Data augmentation pensada para a câmera real: mosaic/scale (mão menor e em outras posições,
  já que na base a mão ocupa quase todo o quadro), variação de cor/brilho (iluminação),
  rotação leve e espelhamento horizontal (sinalizante canhoto/destro).

Uso:
    python src/train.py --data datasets/libras_yolo/data.yaml --epochs 60 --batch 64
"""
import argparse

from ultralytics import YOLO

# Hiperparâmetros compartilhados pelo treino final e pelo k-fold (src/kfold.py)
HYPERPARAMS = dict(
    seed=42,
    patience=15,          # early stopping
    optimizer="AdamW",
    lr0=0.001,
    cos_lr=True,
    warmup_epochs=3,
    # augmentation
    mosaic=1.0,
    close_mosaic=10,      # últimas épocas sem mosaic para refinar em imagens "normais"
    scale=0.5,
    translate=0.1,
    degrees=10,
    fliplr=0.5,
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4,
)


def train(data, model="yolo11n.pt", epochs=60, imgsz=416, batch=64, name="libras_yolo11n",
          project="runs", device="0", freeze=0):
    yolo = YOLO(model)
    yolo.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device, project=project,
               name=name, exist_ok=True, freeze=freeze or None, plots=True, **HYPERPARAMS)
    return yolo.trainer.best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/libras_yolo/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="pesos pré-treinados no COCO (n/s/m)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--freeze", type=int, default=0, help="nº de camadas iniciais congeladas")
    ap.add_argument("--name", default="libras_yolo11n_b64")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    best = train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.name,
                 device=args.device, freeze=args.freeze)
    print("melhor modelo:", best)


if __name__ == "__main__":
    main()
