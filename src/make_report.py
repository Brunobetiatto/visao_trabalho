"""
Gera os gráficos e o arquivo de parâmetros do trabalho em relatorio/.

Gráficos:
  01 distribuição da base             06 k-fold: métricas por fold (média ± desvio)
  02 curvas de treino do modelo final 07 k-fold: mAP50-95 por letra (média ± desvio)
  03 comparação batch 32 x 64         08 k-fold: matriz de confusão somada dos folds
  04 matriz de confusão (teste)       09 exemplos de erros no teste
  05 métricas por letra (teste)       10 distribuição da confiança (acertos x erros)
  + curva precisão-recall e matriz de confusão de detecção do Ultralytics
Texto:
  parametros.txt — todos os parâmetros de pré-processamento, treino, k-fold e ambiente.

Uso:
    python src/make_report.py --final runs/libras_yolo11n_b64 --compare runs/libras_yolo11n_b32
"""
import argparse
import json
import platform
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
METRICS = ("precision", "recall", "mAP50", "mAP50-95", "top1_accuracy")


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def results_csv(run):
    df = pd.read_csv(Path(run) / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    return df


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def plot_training_curves(run, out):
    df = results_csv(run)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for i, (loss, title) in enumerate([("box_loss", "Loss de localização (box)"),
                                       ("cls_loss", "Loss de classificação (cls)"),
                                       ("dfl_loss", "Loss de distribuição (DFL)")]):
        ax[0, i].plot(df.epoch, df[f"train/{loss}"], label="treino")
        ax[0, i].plot(df.epoch, df[f"val/{loss}"], label="validação")
        ax[0, i].set_title(title); ax[0, i].set_xlabel("época"); ax[0, i].legend()
    ax[1, 0].plot(df.epoch, df["metrics/precision(B)"], label="precisão")
    ax[1, 0].plot(df.epoch, df["metrics/recall(B)"], label="recall")
    ax[1, 0].set_title("Precisão e recall (validação)")
    ax[1, 1].plot(df.epoch, df["metrics/mAP50(B)"], label="mAP@50")
    ax[1, 1].plot(df.epoch, df["metrics/mAP50-95(B)"], label="mAP@50-95")
    best = df["metrics/mAP50-95(B)"].idxmax()
    ax[1, 1].axvline(df.epoch[best], ls="--", c="gray", label=f"melhor época ({df.epoch[best]})")
    ax[1, 1].set_title("mAP (validação)")
    ax[1, 2].plot(df.epoch, df["lr/pg1"], c="tab:purple", label="lr (pesos)")  # pg0 = bias, com warmup próprio
    ax[1, 2].set_title("Learning rate (warmup + cosseno)")
    for a in ax[1]:
        a.set_xlabel("época"); a.legend()
    fig.suptitle(f"Curvas de treinamento — {Path(run).name}", fontsize=14)
    save(fig, out, "02_curvas_treino.png")


def plot_batch_comparison(runs, tests, out):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    for run in runs:
        df = results_csv(run)
        lbl = Path(run).name.split("_")[-1]
        ax[0].plot(df.epoch, df["metrics/mAP50-95(B)"], label=lbl)
        ax[1].plot(df.epoch, df["val/cls_loss"], label=lbl)
    ax[0].set_title("mAP@50-95 na validação"); ax[1].set_title("Loss de classificação na validação")
    for a in ax[:2]:
        a.set_xlabel("época"); a.legend()
    x = np.arange(len(METRICS)); w = 0.8 / len(tests)
    for i, (run, t) in enumerate(zip(runs, tests)):
        vals = [t[m] for m in METRICS]
        bars = ax[2].bar(x + i * w - 0.4 + w / 2, vals, w, label=Path(run).name.split("_")[-1])
        ax[2].bar_label(bars, fmt="%.3f", fontsize=7, rotation=90, padding=2)
    ax[2].set_xticks(x, METRICS, rotation=20); ax[2].set_ylim(0.5, 1.08)
    ax[2].set_title("Métricas no TESTE"); ax[2].legend(loc="lower right")
    save(fig, out, "03_comparacao_batch.png")


def confusion_from_predictions(pred_csvs, classes):
    labels = list(classes) + ["nenhuma"]
    cm = np.zeros((len(classes), len(labels)), dtype=int)
    for p in pred_csvs:
        df = pd.read_csv(p, keep_default_na=False)
        for t, pr in zip(df["true"], df["pred"]):
            cm[labels.index(t), labels.index(pr)] += 1
    return cm, labels


def plot_confusion(cm, labels, classes, title, out, name):
    fig, ax = plt.subplots(1, 2, figsize=(20, 8.5))
    norm = cm / cm.sum(1, keepdims=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=classes, ax=ax[0],
                cbar=False, mask=cm == 0, linewidths=0.5, linecolor="#eee")
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=classes, ax=ax[1],
                vmin=0, vmax=1, mask=cm == 0, linewidths=0.5, linecolor="#eee")
    for a, t in zip(ax, ("contagem", "normalizada por linha (recall por letra)")):
        a.set_xlabel("predito"); a.set_ylabel("verdadeiro"); a.set_title(t); a.grid(False)
    acc = np.trace(cm[:, :len(classes)]) / cm.sum()
    fig.suptitle(f"{title} — acurácia top-1 = {acc:.3%}", fontsize=14)
    save(fig, out, name)


def plot_per_class(test, out):
    pc = test["per_class"]
    classes = list(pc)
    fig, ax = plt.subplots(figsize=(15, 4.8))
    x = np.arange(len(classes)); w = 0.27
    for i, (k, lbl) in enumerate([("P", "precisão"), ("R", "recall"), ("mAP50-95", "mAP@50-95")]):
        ax.bar(x + (i - 1) * w, [pc[c][k] for c in classes], w, label=lbl)
    ax.set_xticks(x, classes); ax.set_ylim(0, 1.05); ax.legend(loc="lower left")
    ax.set_title("Métricas por letra no conjunto de TESTE")
    save(fig, out, "05_metricas_por_letra.png")


def plot_kfold(kdir, classes, out):
    folds = sorted(kdir.glob("fold*/fold_metrics.json"))
    if not folds:
        print("   (k-fold ainda não concluído — gráficos 06-08 pulados)")
        return None
    res = [load_json(p) for p in folds]
    k = len(res)

    fig, ax = plt.subplots(1, 2, figsize=(17, 5), sharey=True)
    for a, split in zip(ax, ("val", "test")):
        x = np.arange(len(METRICS)); w = 0.8 / (k + 1)
        for i, r in enumerate(res):
            a.bar(x + i * w - 0.4 + w / 2, [r[split][m] for m in METRICS], w, alpha=0.55, label=f"fold {r['fold']}")
        mu = [np.mean([r[split][m] for r in res]) for m in METRICS]
        sd = [np.std([r[split][m] for r in res], ddof=1) for m in METRICS]
        bars = a.bar(x + k * w - 0.4 + w / 2, mu, w, yerr=sd, capsize=4, color="black", label="média ± dp")
        a.bar_label(bars, labels=[f"{m:.3f}\n±{s:.3f}" for m, s in zip(mu, sd)], fontsize=7, padding=3)
        a.set_xticks(x, METRICS, rotation=15); a.set_ylim(0.5, 1.1)
        a.set_title({"val": "Validação do fold (dados não vistos no treino do fold)",
                     "test": "Conjunto de TESTE fixo"}[split])
    ax[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Validação cruzada {k}-fold agrupada e estratificada", fontsize=14)
    save(fig, out, "06_kfold_metricas.png")

    fig, ax = plt.subplots(figsize=(15, 4.8))
    vals = np.array([[r["val"]["per_class"][c]["mAP50-95"] for c in classes] for r in res])
    ax.bar(classes, vals.mean(0), yerr=vals.std(0, ddof=1), capsize=4, color="tab:blue", alpha=0.8)
    for i in range(k):
        ax.scatter(classes, vals[i], s=12, c="black", zorder=3)
    ax.set_ylim(0, 1.05); ax.set_title(f"mAP@50-95 por letra na validação dos {k} folds (barra = média ± dp, pontos = folds)")
    save(fig, out, "07_kfold_por_letra.png")

    cm, labels = confusion_from_predictions(sorted(kdir.glob("fold*_val/predictions.csv")), classes)
    plot_confusion(cm, labels, classes, f"Matriz de confusão somada das validações dos {k} folds", out,
                   "08_kfold_matriz_confusao.png")
    return res


def plot_errors(pred_csv, out, n=24):
    df = pd.read_csv(pred_csv, keep_default_na=False)
    err = df[df["true"] != df["pred"]]
    if err.empty:
        return
    err = err.sample(min(n, len(err)), random_state=0)
    cols = 6
    rows = int(np.ceil(len(err) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 3.0), squeeze=False)
    for a in np.ravel(ax):
        a.axis("off")
    for a, (_, r) in zip(np.ravel(ax), err.iterrows()):
        a.imshow(cv2.cvtColor(cv2.imread(r["image"]), cv2.COLOR_BGR2RGB))
        a.set_title(f"real {r['true']} → pred {r['pred']}\nconf {float(r['conf']):.2f}", fontsize=9, color="crimson")
    fig.suptitle(f"Erros no TESTE ({len(df[df['true'] != df['pred']])} de {len(df)} imagens)", fontsize=13)
    save(fig, out, "09_erros_teste.png")


def plot_confidence(pred_csv, out):
    df = pd.read_csv(pred_csv, keep_default_na=False)
    ok = df[df["true"] == df["pred"]]["conf"].astype(float)
    bad = df[df["true"] != df["pred"]]["conf"].astype(float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(0, 1, 26)
    ax.hist(ok, bins, alpha=0.7, label=f"acertos ({len(ok)})")
    ax.hist(bad, bins, alpha=0.8, label=f"erros ({len(bad)})", color="crimson")
    ax.set_yscale("log"); ax.set_xlabel("confiança da detecção"); ax.set_ylabel("nº de imagens (log)")
    ax.set_title("Confiança das predições no TESTE"); ax.legend()
    save(fig, out, "10_confianca.png")


def write_params(final, compare, kdir, data_yaml, out):
    import torch
    import ultralytics
    stats = load_json(Path(data_yaml).parent / "stats.json")
    L = []
    sec = lambda t: L.extend(["", "=" * 70, t, "=" * 70])
    L.append("PARÂMETROS DO TRABALHO — Reconhecimento de letras de Libras com YOLO (fine-tuning)")
    L.append(f"Gerado em: {datetime.now():%Y-%m-%d %H:%M}")

    sec("1. AMBIENTE")
    L += [f"Sistema operacional : {platform.platform()}",
          f"Python              : {platform.python_version()}",
          f"PyTorch             : {torch.__version__} (CUDA {torch.version.cuda})",
          f"Ultralytics         : {ultralytics.__version__}",
          f"OpenCV              : {cv2.__version__}",
          f"GPU                 : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"]

    sec("2. BASE DE DADOS E PRÉ-PROCESSAMENTO (src/prepare_dataset.py)")
    p = stats.get("params", {})
    L += [f"Base original       : {p.get('src')}",
          "Formato original    : Pascal VOC (.xml), imagens 200x200 RGB, 1 mão por imagem",
          f"Classes ({len(stats['classes'])})        : {' '.join(stats['classes'])}",
          "Conversão           : (xmin,ymin,xmax,ymax) px -> (classe, xc, yc, w, h) normalizados [0,1]",
          "Limpeza            : pares imagem/xml incompletos descartados; caixas recortadas aos limites",
          "                      da imagem; caixas < 2 px descartadas; rótulos padronizados em maiúsculas",
          f"Arquivos descartados: {stats['skipped_no_xml']} (+ 2 xml sem imagem: C/C145, C/C1449)",
          f"Split               : treino {1 - p['val'] - p['test']:.0%} / validação {p['val']:.0%} / teste {p['test']:.0%}",
          f"Tipo de split       : estratificado por letra, em blocos de {p.get('block')} frames consecutivos",
          "                      (evita vazamento entre frames quase idênticos do mesmo vídeo)",
          f"Seed                : {p.get('seed')}"]
    for s, c in stats["counts"].items():
        L.append(f"  {s:5s} {sum(c.values()):5d} imgs : " + ", ".join(f"{k}={v}" for k, v in c.items()))

    for title, run in (("3. TREINO DO MODELO FINAL", final), ("4. TREINO DE COMPARAÇÃO", compare)):
        if not run or not (Path(run) / "args.yaml").exists():
            continue
        args = yaml.safe_load((Path(run) / "args.yaml").read_text(encoding="utf-8"))
        df = results_csv(run)
        sec(f"{title} — {Path(run).name}")
        L.append(f"Épocas executadas   : {int(df.epoch.max())}   |   tempo total: {df.time.max() / 60:.1f} min")
        L.append(f"Melhor época (val mAP50-95): {int(df.epoch[df['metrics/mAP50-95(B)'].idxmax()])}")
        L.append("Data augmentation efetiva em detecção: mosaic, scale, translate, degrees, fliplr, hsv_h/s/v")
        L.append("  (no Ultralytics, 'erasing' e 'auto_augment' só se aplicam a classificação — sem efeito aqui)")
        L.append("Todos os argumentos do Ultralytics (args.yaml):")
        L += [f"  {k:18s}: {v}" for k, v in args.items()]

    if kdir and (kdir / "kfold_config.json").exists():
        cfg = load_json(kdir / "kfold_config.json")
        sec("5. VALIDAÇÃO CRUZADA K-FOLD (src/kfold.py)")
        L += [f"k                   : {cfg['k']}",
              "Dados               : treino + validação (o TESTE fica de fora e é avaliado em cada fold)",
              f"Agrupamento         : blocos de {cfg['block']} frames consecutivos nunca são separados entre folds",
              "Estratificação      : blocos de cada letra distribuídos entre os folds",
              f"Tamanho dos folds   : {cfg['fold_sizes']}",
              f"Épocas por fold     : {cfg['epochs']}", f"Batch               : {cfg['batch']}",
              f"imgsz               : {cfg['imgsz']}", f"Modelo base         : {cfg['model']}",
              f"Seed                : {cfg['seed']}",
              "Hiperparâmetros     : mesmos do treino final (HYPERPARAMS em src/train.py):"]
        L += [f"  {k:18s}: {v}" for k, v in cfg["hyperparams"].items()]
        if (kdir / "kfold_summary.json").exists():
            sm = load_json(kdir / "kfold_summary.json")
            L.append("Resultado (média ± desvio padrão):")
            for split in ("val", "test"):
                L.append(f"  {split:5s}: " + "  ".join(f"{m}={mu:.4f}±{sd:.4f}" for m, (mu, sd) in sm[split].items()))

    sec("6. AVALIAÇÃO E PROTÓTIPO")
    L += ["Avaliação (src/evaluate.py): imgsz 416, conf 0.25 (top-1), IoU NMS padrão do Ultralytics (0.7)",
          "  métricas de detecção: precisão, recall, mAP@50, mAP@50-95 (geral e por letra)",
          "  métrica de classificação: acurácia top-1 (letra da detecção mais confiante)",
          "Tempo real (src/realtime.py): webcam 1280x720 espelhada, imgsz 416, conf 0.5,",
          "  modo ROI (quadrado central = 80% do menor lado), votação por maioria em 10 frames,",
          "  letra escrita após 1,0 s estável"]
    for run in (final, compare):
        t = Path(f"{run}_test") / "metrics.json"
        if t.exists():
            m = load_json(t)
            L.append(f"Teste {Path(run).name}: " + "  ".join(f"{k}={m[k]:.4f}" for k in METRICS))

    (out / "parametros.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("   parametros.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="runs/libras_yolo11n_b64")
    ap.add_argument("--compare", default="runs/libras_yolo11n_b32")
    ap.add_argument("--kfold", default="runs/kfold")
    ap.add_argument("--data", default="datasets/libras_yolo/data.yaml")
    ap.add_argument("--out", default="relatorio")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(exist_ok=True)
    final, compare, kdir = Path(args.final), Path(args.compare), Path(args.kfold)
    test_dir = Path(f"{final}_test")
    test = load_json(test_dir / "metrics.json")
    classes = test["class_names"]

    print("gerando gráficos em", out)
    shutil.copy(Path(args.data).parent / "dataset_stats.png", out / "01_distribuicao_dataset.png")
    plot_training_curves(final, out)
    if (Path(f"{compare}_test") / "metrics.json").exists():
        plot_batch_comparison([compare, final], [load_json(Path(f"{compare}_test") / "metrics.json"), test], out)
    cm, labels = confusion_from_predictions([test_dir / "predictions.csv"], classes)
    plot_confusion(cm, labels, classes, f"Matriz de confusão no TESTE — {final.name}", out, "04_matriz_confusao_teste.png")
    plot_per_class(test, out)
    plot_kfold(kdir, classes, out)
    plot_errors(test_dir / "predictions.csv", out)
    plot_confidence(test_dir / "predictions.csv", out)
    for src, dst in (("PR_curve.png", "11_curva_precisao_recall.png"),
                     ("confusion_matrix_normalized.png", "12_matriz_confusao_deteccao_ultralytics.png")):
        if (test_dir / src).exists():
            shutil.copy(test_dir / src, out / dst)
    write_params(final, compare, kdir, args.data, out)


if __name__ == "__main__":
    main()
