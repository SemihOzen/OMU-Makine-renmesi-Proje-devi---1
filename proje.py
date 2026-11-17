import argparse
import csv
import json
from pathlib import Path

import numpy as np

from grafikler import make_plots
from metrikler import binary_metrics, roc_curve_auc
from modeller import DogrusalRegresyon, EgitimAyarlari, LojistikRegresyon
from veri_isleme import (
    FEATURE_NAMES, PROJECT_ROOT, StandardScaler, block_features, load_images,
    stratified_split,
)


def write_json(path, value):
    text = json.dumps(value, indent=2, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def class_counts(y):
    return {str(label): int(np.sum(y == label)) for label in (0, 1)}


def fit_and_measure(model, splits, threshold):
    print(f"{model.name.capitalize()} regresyon eğitiliyor...", flush=True)
    X_train, y_train = splits["egitim"]
    model.fit(X_train, y_train)

    split_metrics = {}
    for part, (features, labels) in splits.items():
        scores = model.predict_score(features)
        predicted_labels = (scores >= threshold).astype(int)
        metrics = binary_metrics(labels, predicted_labels)
        _, _, _, auc = roc_curve_auc(labels, scores)
        metrics["roc_auc"] = auc
        split_metrics[part] = metrics

        if part == "test":
            test_scores = scores

    return split_metrics, test_scores


def run_experiment(output_dir=None, seed=42, test_ratio=1 / 3, threshold=0.5, settings=None,
                   *, learning_rate=None, max_iter=50000, tolerance=1e-4):
    output = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "sonuclar"
    for folder in (output, output / "grafikler"):
        folder.mkdir(parents=True, exist_ok=True)

    images, y, original_train_count = load_images()
    X = block_features(images)
    row_ids = np.arange(1, len(y) + 1)
    train_idx, test_idx = stratified_split(y, original_train_count, test_ratio, seed)
    original_test = np.arange(original_train_count, len(y))
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    X_original_test = scaler.transform(X[original_test])

    if settings is None:
        if learning_rate is None:
            design = np.column_stack([X_train, np.ones(len(y_train))])
            hessian = design.T @ design / len(y_train)
            learning_rate = 1.0 / float(np.linalg.eigvalsh(hessian)[-1])
        settings = EgitimAyarlari(learning_rate, max_iter, tolerance)

    splits = {
        "egitim": (X_train, y_train),
        "test": (X_test, y_test),
        "ozgun_test": (X_original_test, y[original_test]),
    }
    models = {
        "dogrusal": DogrusalRegresyon(settings),
        "lojistik": LojistikRegresyon(settings),
    }
    records = []
    results = {}
    official_results = {}
    test_scores = {}
    model_details = {}
    for name, model in models.items():
        metrics, scores = fit_and_measure(model, splits, threshold)
        results[name] = metrics["test"]
        official_results[name] = metrics["ozgun_test"]
        test_scores[name] = scores
        for part, values in metrics.items():
            records.append({"model": name, "veri": part, **values})

        model_details[name] = {
            "weights_standardized": model.w.tolist(),
            "bias_standardized": model.b,
            "n_iter": model.n_iter,
            "stop_reason": model.stop_reason,
            "final_training_loss": model.final_loss,
            "gradient_norm": model.gradient_norm,
        }

    majority_class = int(np.sum(y_train == 1) > np.sum(y_train == 0))
    baseline = binary_metrics(y_test, np.full(len(y_test), majority_class))
    baseline["roc_auc"] = 0.5
    records.append({"model": "cogunluk_referansi", "veri": "test", **baseline})
    fields = ["model", "veri", "accuracy", "precision", "recall", "f1", "specificity",
              "balanced_accuracy", "tn", "fp", "fn", "tp", "n", "roc_auc"]
    write_csv(output / "metrikler.csv", records, fields)

    linear_scores = test_scores["dogrusal"]
    logistic_scores = test_scores["lojistik"]
    prediction_rows = []
    for row, label, linear, logistic in zip(row_ids[test_idx], y_test, linear_scores, logistic_scores):
        prediction_rows.append({
            "original_row": int(row),
            "gercek_sinif": int(label),
            "dogrusal_skor": float(linear),
            "dogrusal_tahmin": int(linear >= threshold),
            "lojistik_olasilik": float(logistic),
            "lojistik_tahmin": int(logistic >= threshold),
        })
    write_csv(output / "test_tahminleri.csv", prediction_rows, list(prediction_rows[0]))

    preprocessing = {
        "feature_order": FEATURE_NAMES,
        "mean_train": scaler.mean.tolist(),
        "std_train_ddof0": scaler.std.tolist(),
    }
    inference_bundle = {
        "threshold": threshold,
        "positive_class": 1,
        "preprocessing": preprocessing,
        "models": model_details,
    }
    write_json(output / "model_parametreleri.json", inference_bundle)

    summary = {
        "dataset": {
            "name": "MNIST (0 ve 1)",
            "raw_samples": len(y),
            "raw_class_counts": class_counts(y),
            "n_features": X.shape[1],
            "image_shape": [28, 28],
        },
        "split": {
            "seed": seed,
            "requested_test_ratio": test_ratio,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "train_class_counts": class_counts(y_train),
            "test_class_counts": class_counts(y_test),
            "original_mnist_test_count": len(original_test),
        },
        "settings": {
            "learning_rate": settings.learning_rate,
            "max_iter": settings.max_iter,
            "gradient_tolerance": settings.tolerance,
            "decision_threshold": threshold,
        },
        "models": model_details,
        "test_metrics": results,
        "original_mnist_test_metrics": official_results,
        "baseline": {"predicted_class": majority_class, **baseline},
    }
    write_json(output / "ozet.json", summary)
    examples = [original_test[np.flatnonzero(y[original_test] == label)[0]] for label in (0, 1)]
    make_plots(output / "grafikler", models, images[examples], X[examples], X_test, y_test, results, threshold)

    print_console_summary(summary)
    return summary


def print_console_summary(summary):
    dataset = summary["dataset"]
    split = summary["split"]
    threshold = summary["settings"]["decision_threshold"]
    lines = [
        "MNIST 0/1 - İKİ REGRESYON KARŞILAŞTIRMASI",
        f"Toplam görüntü: {dataset['raw_samples']} | Blok özellikleri: {dataset['n_features']}",
        f"Eğitim: {split['n_train']} | Test: {split['n_test']} | Seed: {split['seed']} | Eşik: {threshold}",
        f"Özgün MNIST test alt kümesi: {split['original_mnist_test_count']} görüntü",
        "",
        f"{'Model':<22} {'Accuracy':>10} {'Recall 0':>10} {'Recall 1':>10} {'F1':>10} {'ROC AUC':>10}",
    ]
    test_metrics = {**summary["test_metrics"], "cogunluk_referansi": summary["baseline"]}
    for name, metrics in test_metrics.items():
        lines.append(
            f"{name:<22} {metrics['accuracy']:>10.4f} {metrics['specificity']:>10.4f} "
            f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['roc_auc']:>10.6f}"
        )

    lines.extend(["", "Eğitim durumu:"])
    for name, details in summary["models"].items():
        lines.append(
            f"{name}: {details['n_iter']} adım, {details['stop_reason']}, "
            f"||gradyan||={details['gradient_norm']:.3g}"
        )

    lines.extend(["", "Özgün MNIST test alt kümesinde doğruluk:"])
    for name, metrics in summary["original_mnist_test_metrics"].items():
        lines.append(f"{name}: {metrics['accuracy']:.6f}")

    print("\n".join(lines))


def main():
    defaults = EgitimAyarlari()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "sonuclar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=1/3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=None, help="Verilmezse yalnız eğitimden güvenli ortak adım hesaplanır.")
    parser.add_argument("--max-iter", type=int, default=50000)
    parser.add_argument("--tolerance", type=float, default=defaults.tolerance)
    args = parser.parse_args()
    run_experiment(args.output_dir, args.seed, args.test_ratio, args.threshold,
                   learning_rate=args.learning_rate, max_iter=args.max_iter, tolerance=args.tolerance)


if __name__ == "__main__":
    main()
