"""
Pré-processamento: converte a base de Libras (Pascal VOC .xml) para o formato YOLO.

- Lê cada par imagem/.xml, descarta arquivos órfãos e caixas inválidas.
- Converte (xmin, ymin, xmax, ymax) em (classe, x_centro, y_centro, largura, altura) normalizados.
- Faz split treino/val/teste ESTRATIFICADO POR CLASSE e em BLOCOS CONTÍGUOS de frames:
  as imagens são frames sequenciais de vídeo (quase duplicadas), então um split aleatório
  colocaria frames vizinhos em treino e teste e inflaria as métricas (data leakage).
- Gera data.yaml e um resumo estatístico (stats.json + gráfico de distribuição).

Uso:
    python src/prepare_dataset.py --src "C:/Users/Casa/Downloads/k4gs3bmx5k-3" --out datasets/libras_yolo
"""
import argparse
import json
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def frame_index(stem: str) -> int:
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name").strip().upper()
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        # recorta para dentro da imagem e descarta caixas degeneradas
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        boxes.append((name, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))
    return boxes


def block_split(items, ratios, block_size, rng):
    """Agrupa frames consecutivos em blocos e sorteia blocos (não frames) para cada split."""
    items = sorted(items, key=lambda p: frame_index(p.stem))
    blocks = [items[i:i + block_size] for i in range(0, len(items), block_size)]
    rng.shuffle(blocks)
    n = len(blocks)
    n_test = max(1, round(n * ratios[2]))
    n_val = max(1, round(n * ratios[1]))
    split = {"test": blocks[:n_test], "val": blocks[n_test:n_test + n_val], "train": blocks[n_test + n_val:]}
    return {k: [p for b in v for p in b] for k, v in split.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="pasta da base original (uma subpasta por letra)")
    ap.add_argument("--out", default="datasets/libras_yolo")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--block", type=int, default=20, help="tamanho do bloco de frames consecutivos")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out).resolve()
    rng = random.Random(args.seed)
    if out.exists():
        shutil.rmtree(out)

    # 1) coleta pares válidos
    per_class = defaultdict(list)
    skipped = []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        for img in d.glob("*.jpg"):
            xml = img.with_suffix(".xml")
            if not xml.exists():
                skipped.append(str(img.relative_to(src)))
                continue
            per_class[d.name.upper()].append(img)
    classes = sorted(per_class)
    cls_id = {c: i for i, c in enumerate(classes)}

    # 2) split + conversão
    stats = {"params": vars(args), "classes": classes, "skipped_no_xml": skipped, "counts": {}, "box_wh": []}
    counts = defaultdict(Counter)
    for c in classes:
        splits = block_split(per_class[c], (1 - args.val - args.test, args.val, args.test), args.block, rng)
        for split, imgs in splits.items():
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            for img in imgs:
                boxes = parse_voc(img.with_suffix(".xml"))
                if not boxes:
                    skipped.append(str(img.relative_to(src)))
                    continue
                shutil.copy2(img, out / "images" / split / img.name)
                with open(out / "labels" / split / f"{img.stem}.txt", "w") as f:
                    for name, xc, yc, bw, bh in boxes:
                        f.write(f"{cls_id[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                        stats["box_wh"].append((bw, bh))
                counts[split][c] += 1
    stats["counts"] = {s: dict(counts[s]) for s in ("train", "val", "test")}

    # 3) data.yaml
    yaml = [f"path: {out.as_posix()}", "train: images/train", "val: images/val", "test: images/test",
            f"nc: {len(classes)}", "names:"] + [f"  {i}: {c}" for i, c in enumerate(classes)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    # 4) resumo + gráficos
    wh = stats.pop("box_wh")
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bottom = [0] * len(classes)
    for s in ("train", "val", "test"):
        vals = [counts[s][c] for c in classes]
        ax[0].bar(classes, vals, bottom=bottom, label=s)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax[0].set_title("Imagens por letra e split"); ax[0].legend()
    ax[1].scatter([w for w, _ in wh], [h for _, h in wh], s=3, alpha=0.3)
    ax[1].set_xlabel("largura (norm.)"); ax[1].set_ylabel("altura (norm.)"); ax[1].set_title("Tamanho das caixas")
    fig.tight_layout(); fig.savefig(out / "dataset_stats.png", dpi=120)

    for s in ("train", "val", "test"):
        print(f"{s:5s}: {sum(counts[s].values()):5d} imagens  {dict(counts[s])}")
    print(f"ignorados: {skipped}")
    print(f"data.yaml -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
