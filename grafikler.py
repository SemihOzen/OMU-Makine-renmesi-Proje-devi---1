from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from metrikler import roc_curve_auc

COLORS = ["#167D9A", "#D46A35"]
LABELS = {"dogrusal": "Doğrusal regresyon", "lojistik": "Lojistik regresyon"}


def save(fig, folder, name):
    fig.savefig(folder / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_plots(folder, models, example_images, example_blocks, X_test, y_test, results, threshold):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(2, 2, figsize=(7, 6), constrained_layout=True)
    for label in (0, 1):
        axes[label, 0].imshow(example_images[label], cmap="gray", vmin=0, vmax=255)
        axes[label, 1].imshow(example_blocks[label].reshape(7, 7), cmap="gray", vmin=0, vmax=1)
        axes[label, 0].set_ylabel(f"Gerçek etiket {label}")
        for ax in axes[label]:
            ax.set(xticks=[], yticks=[])
    axes[0, 0].set_title("MNIST görüntüsü (28x28)")
    axes[0, 1].set_title("Blok ortalamaları (7x7)")
    fig.suptitle("Örnek Görüntüler ve 4x4 Blok Özellikleri")
    save(fig, folder, "01_goruntu_ve_bloklar.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (name, model), color in zip(axes, models.items(), COLORS):
        history = np.asarray(model.history)
        ax.plot(history[:, 0], history[:, 1], color=color, linewidth=2)
        ax.set(title=LABELS[name], xlabel="İterasyon", ylabel="Eğitim kaybı")
        ax.set_xscale("symlog", linthresh=10)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("MSE / 2")
    axes[1].set_ylabel("Binary Cross-Entropy")
    fig.suptitle("Eğitim Kaybı Değişimi", fontsize=14)
    save(fig, folder, "02_egitim_kayiplari.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    vmax = int(np.bincount(y_test).max())
    for ax, name in zip(axes, models):
        metrics = results[name]
        matrix = np.array([
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        for actual in (0, 1):
            for predicted in (0, 1):
                count = matrix[actual, predicted]
                text_color = "white" if count > vmax / 2 else "#182C3C"
                ax.text(predicted, actual, str(count), ha="center", va="center",
                        fontsize=19, color=text_color)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
               xlabel="Tahmin edilen sınıf", ylabel="Gerçek sınıf", title=LABELS[name])
    fig.suptitle(f"Hata Matrisleri (Test kümesi: {len(y_test)} örnek, eşik: {threshold:g})", fontsize=14)
    save(fig, folder, "03_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    displayed_metrics = [
        ("accuracy", "Accuracy"),
        ("specificity", "Specificity"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC AUC"),
    ]
    positions = np.arange(len(displayed_metrics))
    for index, (name, color) in enumerate(zip(models, COLORS)):
        values = [results[name][key] for key, _ in displayed_metrics]
        bar_positions = positions + (index - 0.5) * 0.34
        bars = ax.bar(bar_positions, values, width=0.34, color=color, label=LABELS[name])
        labels = [f"{value:.6f}" if key == "roc_auc" else f"{value:.4f}"
                  for (key, _), value in zip(displayed_metrics, values)]
        ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.set(xticks=positions, xticklabels=[label for _, label in displayed_metrics],
           ylim=(0, 1.16), ylabel="Skor", title="Test Kümesi Metrik Karşılaştırması")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.15)
    save(fig, folder, "04_metrik_karsilastirmasi.png")

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for (name, model), color in zip(models.items(), COLORS):
        fpr, tpr, _, auc = roc_curve_auc(y_test, model.predict_score(X_test))
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{LABELS[name]} (AUC={auc:.6f})")
    ax.plot([0, 1], [0, 1], "--", color="#777777", label="Rastgele tahmin")
    ax.set(xlabel="Yanlış Pozitif Oranı (FPR)", ylabel="Doğru Pozitif Oranı (TPR)",
           title="Test ROC Eğrileri", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    save(fig, folder, "05_roc_egri.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, (name, model) in zip(axes, models.items()):
        scores = model.predict_score(X_test)
        if name == "dogrusal":
            bins = np.linspace(min(0, scores.min()), max(1, scores.max()), 41)
        else:
            bins = np.linspace(0, 1, 26)
        for label, color in enumerate(COLORS):
            selected = scores[y_test == label]
            weights = np.full(len(selected), 1 / len(selected))
            ax.hist(selected, bins=bins, weights=weights, alpha=0.6, color=color,
                    label=f"Sınıf {label}")
        ax.axvline(threshold, color="#222222", linestyle="--", label=f"Eşik = {threshold:g}")
        ax.set(title=LABELS[name], xlabel="Tahmin skoru" if name == "dogrusal" else "Tahmin olasılığı",
               ylabel="Oran")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Test Kümesi Tahmin Skoru Dağılımları", fontsize=13)
    save(fig, folder, "06_test_skorlari.png")
