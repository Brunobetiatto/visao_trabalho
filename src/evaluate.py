"""
Avaliação de um modelo treinado em um split (por padrão o TESTE, nunca visto no treino/validação).

Gera em runs/<nome>_<split>/:
- métricas de detecção (precision, recall, mAP50, mAP50-95) gerais e por letra;
- matriz de confusão de detecção e curvas PR/F1 (Ultralytics);
- acurácia "de classificação" top-1: a letra da detecção mais confiante vs. a letra verdadeira;
- metrics.json (resumo + matriz de confusão) e predictions.csv (predição por imagem).

Uso:
    python src/evaluate.py --weights runs/libras_yolo11n_b64/weights/best.pt
"""
import argparse
import csv
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

NO_DET = -1  # imagem em que o modelo não detectou nenhuma mão


def split_images(data, split):
    """Imagens de um split do data.yaml: uma pasta ou uma lista .txt (usada no k-fold)."""
    cfg = yaml.safe_load(Path(data).read_text(encoding="utf-8"))
    src = Path(cfg.get("path", Path(data).parent)) / cfg[split]
    if src.suffix == ".txt":
        return [Path(line) for line in src.read_text(encoding="utf-8").split()]
    return sorted(src.glob("*.jpg"))


def label_path(img):
    """Convenção YOLO: .../images/<split>/x.jpg -> .../labels/<split>/x.txt"""
    return img.parents[2] / "labels" / img.parent.name / f"{img.stem}.txt"


def top1_predictions(model, image_paths, imgsz, conf=0.25):
    """Para cada imagem: (caminho, classe verdadeira, classe prevista ou NO_DET, confiança)."""
    rows = []
    for res in model.predict([str(p) for p in image_paths], imgsz=imgsz, conf=conf, verbose=False, stream=True):
        img = Path(res.path)
        gt = int(label_path(img).read_text().split()[0])
        if len(res.boxes) == 0:
            rows.append((str(img), gt, NO_DET, 0.0))
        else:
            i = int(res.boxes.conf.argmax())
            rows.append((str(img), gt, int(res.boxes.cls[i]), float(res.boxes.conf[i])))
    return rows


def evaluate(weights, data, imgsz=416, split="test", project="runs", name=None):
    model = YOLO(weights)
    name = name or f"{Path(weights).parents[1].name}_{split}"
    m = model.val(data=data, split=split, imgsz=imgsz, project=project, name=name, exist_ok=True, plots=True)
    out_dir = Path(project) / name

    names = model.names
    per_class = {names[int(c)]: {"P": float(p), "R": float(r), "mAP50": float(a50), "mAP50-95": float(a)}
                 for c, p, r, a50, a in zip(m.box.ap_class_index, m.box.p, m.box.r, m.box.ap50, m.box.ap)}

    rows = top1_predictions(model, split_images(data, split), imgsz)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "true", "pred", "conf"])
        w.writerows((p, names[t], names.get(pr, "nenhuma"), f"{c:.4f}") for p, t, pr, c in rows)

    summary = {
        "weights": str(weights), "split": split,
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
        "top1_accuracy": sum(t == p for _, t, p, _ in rows) / len(rows),
        "images_without_detection": sum(p == NO_DET for *_, p, _ in rows),
        "n_images": len(rows),
        "per_class": per_class,
        "class_names": [names[i] for i in range(len(names))],
        "detection_confusion_matrix": m.confusion_matrix.matrix.tolist(),  # [pred][true], última linha/col = fundo
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/libras_yolo11n_b64/weights/best.pt")
    ap.add_argument("--data", default="datasets/libras_yolo/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=416)
    args = ap.parse_args()

    summary, out_dir = evaluate(args.weights, args.data, args.imgsz, args.split)
    keys = ("precision", "recall", "mAP50", "mAP50-95", "top1_accuracy", "images_without_detection", "n_images")
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    print(f"métricas completas -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
