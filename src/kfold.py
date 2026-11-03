"""
Validação cruzada k-fold (padrão k=5) agrupada e estratificada.

- Usa apenas treino+validação; o conjunto de TESTE fica de fora e é usado só para
  medir cada modelo do fold no fim.
- Agrupada: frames consecutivos (blocos de --block frames) ficam sempre no mesmo fold,
  evitando que frames quase idênticos caiam em treino e validação ao mesmo tempo.
- Estratificada: os blocos de cada letra são distribuídos entre os k folds, então todo
  fold tem todas as letras na mesma proporção.

Para cada fold k: treina em k-1 partes, valida na parte k e avalia também no teste.
Ao final grava runs/kfold/kfold_summary.csv e kfold_summary.json (média ± desvio padrão).
É retomável: folds com metrics.json já existente são pulados.

Uso:
    python src/kfold.py --k 5 --epochs 30 --batch 64
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from evaluate import evaluate, label_path
from prepare_dataset import frame_index
from train import HYPERPARAMS, train

METRICS = ("precision", "recall", "mAP50", "mAP50-95", "top1_accuracy")


def make_folds(data_yaml, k, block, seed):
    cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    names = cfg["names"]
    per_class = defaultdict(list)
    for split in ("train", "val"):
        for img in (root / cfg[split]).glob("*.jpg"):
            per_class[int(label_path(img).read_text().split()[0])].append(img)

    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    offset = 0  # continua a rotação entre letras para os folds terem tamanhos parecidos
    for c in sorted(per_class):
        imgs = sorted(per_class[c], key=lambda p: frame_index(p.stem))
        blocks = [imgs[i:i + block] for i in range(0, len(imgs), block)]
        rng.shuffle(blocks)
        for i, b in enumerate(blocks):
            folds[(offset + i) % k].extend(b)
        offset += len(blocks)
    return root, names, folds


def write_fold_yaml(root, names, folds, k, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    val = folds[k]
    train_imgs = [p for i, f in enumerate(folds) if i != k for p in f]
    (out_dir / "train.txt").write_text("\n".join(p.as_posix() for p in train_imgs), encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(p.as_posix() for p in val), encoding="utf-8")
    cfg = {"path": out_dir.resolve().as_posix(), "train": "train.txt", "val": "val.txt",
           "test": (root / "images" / "test").as_posix(), "nc": len(names), "names": names}
    data = out_dir / "data.yaml"
    data.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data, len(train_imgs), len(val)


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/libras_yolo/data.yaml")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--project", default="runs/kfold")
    args = ap.parse_args()

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    root, names, folds = make_folds(args.data, args.k, args.block, args.seed)
    config = {**vars(args), "fold_sizes": [len(f) for f in folds], "hyperparams": HYPERPARAMS}
    (project / "kfold_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    results = []
    for k in range(args.k):
        fold_dir = project / f"fold{k + 1}"
        done = fold_dir / "fold_metrics.json"
        if done.exists():
            print(f"fold {k + 1}: já concluído, pulando")
            results.append(json.loads(done.read_text(encoding="utf-8")))
            continue

        data, n_train, n_val = write_fold_yaml(root, names, folds, k, Path("datasets/libras_yolo/kfold") / f"fold{k + 1}")
        print(f"\n===== FOLD {k + 1}/{args.k}: {n_train} treino / {n_val} validação =====")
        best = train(str(data), args.model, args.epochs, args.imgsz, args.batch,
                     name=f"fold{k + 1}", project=str(project))
        val_sum, _ = evaluate(best, str(data), args.imgsz, "val", str(project), f"fold{k + 1}_val")
        test_sum, _ = evaluate(best, str(data), args.imgsz, "test", str(project), f"fold{k + 1}_test")
        res = {"fold": k + 1, "n_train": n_train, "n_val": n_val, "val": val_sum, "test": test_sum}
        done.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(res)

    # agregação
    lines = ["fold,split," + ",".join(METRICS)]
    for r in results:
        for split in ("val", "test"):
            lines.append(f"{r['fold']},{split}," + ",".join(f"{r[split][m]:.4f}" for m in METRICS))
    summary = {}
    for split in ("val", "test"):
        summary[split] = {m: mean_std([r[split][m] for r in results]) for m in METRICS}
        lines.append(f"media,{split}," + ",".join(f"{summary[split][m][0]:.4f}" for m in METRICS))
        lines.append(f"desvio,{split}," + ",".join(f"{summary[split][m][1]:.4f}" for m in METRICS))
    (project / "kfold_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (project / "kfold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== RESUMO K-FOLD (média ± desvio) =====")
    for split in ("val", "test"):
        print(split.upper(), "  ".join(f"{m}={mu:.4f}±{sd:.4f}" for m, (mu, sd) in summary[split].items()))


if __name__ == "__main__":
    main()
