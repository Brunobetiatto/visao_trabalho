"""
Figuras no formato do pôster (fonte grande, pouco texto, cores do template PUCPR) em relatorio/poster/.
Também grava relatorio/poster/metricas_poster.json com os números da tabela do pôster.

Uso:
    python src/poster_figures.py [--font-dir pasta_com_Montserrat-*.ttf]
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

RED, WINE, PURPLE, GRAY, LIGHT = "#D52128", "#8B1A4A", "#5B3A70", "#6B6B6B", "#F3E8EE"
CMAP = LinearSegmentedColormap.from_list("poster", ["#FFFFFF", "#E9B7C6", WINE, "#4A0D28"])
BOLD = None


def setup_fonts(font_dir):
    global BOLD
    font_dir = Path(font_dir) if font_dir else None
    if font_dir and (font_dir / "Montserrat-Regular.ttf").exists():
        for f in font_dir.glob("Montserrat-*.ttf"):
            fm.fontManager.addfont(str(f))
        plt.rcParams["font.family"] = fm.FontProperties(fname=font_dir / "Montserrat-Regular.ttf").get_name()
        BOLD = fm.FontProperties(fname=font_dir / "Montserrat-Bold.ttf")
    else:
        BOLD = fm.FontProperties(weight="bold")
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": "#BBBBBB", "axes.labelcolor": "#333333",
                         "xtick.color": "#333333", "ytick.color": "#333333", "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300})


def bold(size, color="#111111"):
    p = BOLD.copy()
    p.set_size(size)
    return {"fontproperties": p, "color": color}


def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight", pad_inches=0.04, transparent=False, facecolor="white")
    plt.close(fig)
    print("  ", name)


def read_rgb(path, crop=3):
    """Lê a imagem em RGB, removendo a moldura azul de ~3 px que existe nas fotos da base."""
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return img[crop:-crop, crop:-crop]


def first_image_per_class(data_root, split="train"):
    names = {}
    for lbl in sorted((data_root / "labels" / split).glob("*.txt")):
        c = int(lbl.read_text().split()[0])
        if c not in names:
            names[c] = data_root / "images" / split / f"{lbl.stem}.jpg"
    return names


def fig_letters(data_root, classes, out):
    imgs = first_image_per_class(data_root)
    fig, axes = plt.subplots(3, 5, figsize=(4.45, 2.72))
    for c, ax in enumerate(axes.flat):
        ax.imshow(read_rgb(imgs[c]))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0.06, 0.94, classes[c], transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc=WINE, ec="none"), **bold(11, "white"))
    fig.subplots_adjust(wspace=0.06, hspace=0.06, left=0, right=1, top=1, bottom=0)
    save(fig, out, "p1_letras.png")


def fig_distribution(data_root, classes, out):
    """Imagens por letra, empilhadas por split (mostra o desbalanceamento e a divisão 70/15/15)."""
    counts = json.loads((data_root / "stats.json").read_text(encoding="utf-8"))["counts"]
    fig, ax = plt.subplots(figsize=(4.45, 3.1))
    x = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for split, lbl, color in (("train", "treino", PURPLE), ("val", "validação", WINE), ("test", "teste", RED)):
        v = np.array([counts[split].get(c, 0) for c in classes])
        ax.bar(x, v, 0.75, bottom=bottom, color=color, label=lbl)
        bottom += v
    for xi, total in zip(x, bottom):
        ax.text(xi, total + 8, f"{int(total)}", ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.set_xticks(x, classes, fontsize=11)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, bottom.max() * 1.12)
    ax.set_ylabel("imagens")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=10, loc="upper right", bbox_to_anchor=(1, 1.02))
    ax.set_title("Imagens por letra e split", loc="left", pad=14, **bold(12.5))
    save(fig, out, "p2_distribuicao.png")


def fig_architecture(weights, out):
    from ultralytics import YOLO
    m = YOLO(weights).model
    n = lambda r: sum(p.numel() for i in r for p in m.model[i].parameters()) / 1e6
    det = m.model[-1]
    box_p = sum(p.numel() for p in det.cv2.parameters()) / 1e6
    cls_p = sum(p.numel() for p in det.cv3.parameters()) / 1e6
    total = sum(p.numel() for p in m.parameters()) / 1e6
    from ultralytics.utils.torch_utils import get_flops
    gflops = get_flops(m, 416)
    blocks = [("Entrada", "imagem RGB 416 × 416", None, GRAY),
              ("Backbone", "Conv · C3k2 ×4 · SPPF · C2PSA", n(range(0, 11)), PURPLE),
              ("Neck (FPN + PAN)", "upsample · concat · C3k2 ×4", n(range(11, len(m.model) - 1)), PURPLE)]
    W, H = 4.45, 3.75
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    row, gap = 0.78, 0.2
    y = H - row
    for title, sub, params, color in blocks:
        ax.add_patch(FancyBboxPatch((0.03, y), W - 0.06, row - 0.02, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=color, ec="none"))
        ax.text(0.17, y + row * 0.64, title, va="center", **bold(13, "white"))
        ax.text(0.17, y + row * 0.27, sub, va="center", fontsize=10.5, color="#F1DDE6")
        if params:
            ax.text(W - 0.17, y + row * 0.64, f"{params:.2f} M".replace(".", ","), va="center", ha="right",
                    **bold(12, "white"))
        ax.annotate("", xy=(W / 2, y - gap + 0.01), xytext=(W / 2, y - 0.01),
                    arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.6, mutation_scale=12))
        y -= row + gap
    # cabeça de detecção: ramo de caixas (veio do COCO) e ramo de classes (novo, 15 letras)
    half = (W - 0.06 - 0.12) / 2
    for i, (title, sub, params, color) in enumerate(
            [("Head · caixas", "DFL · 3 escalas", box_p, PURPLE), ("Head · classes", "15 letras (novo)", cls_p, RED)]):
        x = 0.03 + i * (half + 0.12)
        ax.add_patch(FancyBboxPatch((x, y), half, row - 0.02, boxstyle="round,pad=0,rounding_size=0.1",
                                    fc=color, ec="none"))
        ax.text(x + 0.12, y + row * 0.64, title, va="center", **bold(11, "white"))
        ax.text(x + 0.12, y + row * 0.27, f"{sub} · {params:.2f} M".replace(".", ","), va="center",
                fontsize=9.5, color="#F7E6EC")
    y -= 0.42
    ax.add_patch(FancyBboxPatch((0.08, y + 0.08), 0.22, 0.2, boxstyle="round,pad=0,rounding_size=0.04", fc=PURPLE, ec="none"))
    ax.text(0.38, y + 0.18, "pesos do COCO", va="center", fontsize=10.5, color="#333333")
    ax.add_patch(FancyBboxPatch((2.05, y + 0.08), 0.22, 0.2, boxstyle="round,pad=0,rounding_size=0.04", fc=RED, ec="none"))
    ax.text(2.35, y + 0.18, "treinado do zero", va="center", fontsize=10.5, color="#333333")
    y -= 0.45
    ax.text(W / 2, y + 0.2, f"Total: {total:.2f} M parâmetros · {gflops:.1f} GFLOPs (416 px)".replace(".", ","),
            ha="center", va="center", **bold(11.5))
    ax.set_ylim(y, H)
    fig.set_size_inches(W, H - y)
    save(fig, out, "p3_arquitetura.png")
    return total


def fig_confusion(kdir, classes, out):
    labels = list(classes) + ["nenhuma"]
    cm = np.zeros((len(classes), len(labels)))
    for p in sorted(kdir.glob("fold*_val/predictions.csv")):
        df = pd.read_csv(p, keep_default_na=False)
        for t, pr in zip(df["true"], df["pred"]):
            cm[labels.index(t), labels.index(pr)] += 1
    if cm[:, -1].sum() == 0:
        cm, labels = cm[:, :-1], labels[:-1]
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.imshow(norm, cmap=CMAP, vmin=0, vmax=1)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8.5 if i == j else 10,
                        color="white" if v > 0.5 else WINE, fontproperties=BOLD if i != j else None)
    ax.set_xticks(range(len(labels)), labels, fontsize=11)
    ax.set_yticks(range(len(classes)), classes, fontsize=11)
    ax.set_xlabel("predito", fontsize=12); ax.set_ylabel("real", fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Matriz de confusão (%) · validação dos 5 folds", pad=10, **bold(12.5))
    save(fig, out, "p4_confusao_kfold.png")


def fig_f1_curve(weights, data_yaml, out):
    """Curva F1 × limiar de confiança no teste (média das 15 letras), do validador do Ultralytics."""
    from ultralytics import YOLO
    m = YOLO(weights).val(data=data_yaml, split="test", imgsz=416, plots=False, verbose=False,
                          project="runs", name="_tmp_f1", exist_ok=True)
    px, f1 = m.box.px, m.box.f1_curve.mean(0)
    i = int(f1.argmax())
    fig, ax = plt.subplots(figsize=(3.9, 2.5))
    for c in m.box.f1_curve:
        ax.plot(px, c, color="#D9C3CD", lw=0.7)
    ax.plot(px, f1, color=RED, lw=2.4)
    ax.scatter([px[i]], [f1[i]], color=RED, zorder=3, s=30)
    ax.annotate(f"F1 = {f1[i]:.3f}\nconf = {px[i]:.2f}".replace(".", ","), (px[i], f1[i]),
                xytext=(px[i] - 0.05, 0.62), ha="right", fontsize=10.5,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=1))
    ax.set_xlim(0, 1); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("limiar de confiança"); ax.set_ylabel("F1 (teste)")
    ax.grid(alpha=0.25)
    ax.set_title("F1 × confiança", loc="left", **bold(12.5))
    save(fig, out, "p5_f1_confianca.png")
    return float(f1[i]), float(px[i])


def fig_batch_curves(runs, out):
    fig, ax = plt.subplots(figsize=(3.9, 2.5))
    for run, color, lbl in zip(runs, (GRAY, RED), ("batch 32", "batch 64")):
        df = pd.read_csv(run / "results.csv")
        df.columns = [c.strip() for c in df.columns]
        ax.plot(df.epoch, df["metrics/mAP50-95(B)"], color=color, lw=2.2 if "64" in lbl else 1.6, label=lbl)
    ax.set_ylim(0.6, 0.97); ax.set_xlabel("época"); ax.set_ylabel("mAP@50-95 (val)")
    ax.grid(alpha=0.25); ax.legend(frameon=False, loc="lower right", fontsize=11)
    ax.set_title("Batch 32 × 64", loc="left", **bold(12.5))
    save(fig, out, "p6_curvas_batch.png")


def fig_losses(run, out):
    df = pd.read_csv(run / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    fig, ax = plt.subplots(figsize=(4.45, 2.2))
    for k, lbl, color in (("box_loss", "caixa", PURPLE), ("cls_loss", "classe", RED)):
        ax.plot(df.epoch, df[f"train/{k}"], color=color, lw=2, label=f"{lbl} · treino")
        ax.plot(df.epoch, df[f"val/{k}"], color=color, lw=1.6, ls="--", label=f"{lbl} · validação")
    cm = len(df) - 10
    ax.axvline(cm + 0.5, color="#999999", lw=1, ls=":")
    ax.text(cm - 0.5, ax.get_ylim()[1] * 0.97, "mosaic\ndesligado →", ha="right", va="top", fontsize=9.5, color=GRAY)
    ax.set_xlabel("época"); ax.set_ylabel("loss")
    ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.set_title("Losses · AdamW · lr 0,001 · batch 64", loc="left", **bold(12))
    save(fig, out, "p7_losses.png")


def metrics_table(final, kdir, f1_best):
    """Média (± dp entre folds) de cada métrica; F1 = média do F1 de cada letra, como no Ultralytics."""
    keys = ("precision", "recall", "mAP50", "mAP50-95", "top1_accuracy")
    f1 = lambda m: float(np.mean([2 * c["P"] * c["R"] / (c["P"] + c["R"]) if c["P"] + c["R"] else 0.0
                                  for c in m["per_class"].values()]))
    t = json.loads(Path(f"{final}_test/metrics.json").read_text(encoding="utf-8"))
    rows = {"Teste · modelo final": [t]}
    folds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(kdir.glob("fold*/fold_metrics.json"))]
    rows["K-fold 5 · validação"] = [f["val"] for f in folds]
    rows["K-fold 5 · teste"] = [f["test"] for f in folds]
    table = {}
    for name, ms in rows.items():
        vals = {k: [m[k] for m in ms] for k in keys}
        vals["F1"] = [f1(m) for m in ms]
        table[name] = {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) for k, v in vals.items()}
    table["_f1_best_conf"] = f1_best
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font-dir", default=None)
    ap.add_argument("--data", default="datasets/libras_yolo")
    ap.add_argument("--final", default="runs/libras_yolo11n_b64")
    ap.add_argument("--compare", default="runs/libras_yolo11n_b32")
    ap.add_argument("--kfold", default="runs/kfold")
    ap.add_argument("--out", default="relatorio/poster")
    args = ap.parse_args()

    setup_fonts(args.font_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data_root, final, kdir = Path(args.data), Path(args.final), Path(args.kfold)
    data_yaml = str(data_root / "data.yaml")
    weights = str(final / "weights" / "best.pt")
    classes = json.loads(Path(f"{final}_test/metrics.json").read_text(encoding="utf-8"))["class_names"]

    print("figuras do pôster em", out)
    fig_letters(data_root, classes, out)
    fig_distribution(data_root, classes, out)
    total = fig_architecture(weights, out)
    fig_confusion(kdir, classes, out)
    f1_best = fig_f1_curve(weights, data_yaml, out)
    fig_batch_curves([Path(args.compare), final], out)
    fig_losses(final, out)
    table = metrics_table(final, kdir, f1_best)
    table["_params_M"] = total
    (out / "metricas_poster.json").write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    print("   metricas_poster.json")


if __name__ == "__main__":
    main()
