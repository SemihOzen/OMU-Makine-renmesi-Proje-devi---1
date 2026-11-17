from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_PATH = PROJECT_ROOT / 'veri' / 'ham' / 'mnist.npz'
FEATURE_NAMES = [f'blok_{i + 1}_{j + 1}' for i in range(7) for j in range(7)]


def load_images():
    """0 ve 1 görüntülerini, özgün eğitim örnekleri önce gelecek şekilde yükle."""
    with np.load(RAW_PATH, allow_pickle=False) as source:
        train_labels = source['y_train']
        test_labels = source['y_test']
        train_mask = train_labels <= 1
        test_mask = test_labels <= 1

        train_images = source['x_train'][train_mask]
        test_images = source['x_test'][test_mask]
        images = np.concatenate([train_images, test_images])
        labels = np.concatenate([train_labels[train_mask], test_labels[test_mask]])
    return images, labels, len(train_images)


def block_features(images):
    """Her görüntünün 49 blok ortalamasını 0-1 aralığında döndür."""
    blocks = images.reshape(-1, 7, 4, 7, 4)
    averages = blocks.mean(axis=(2, 4))
    return averages.reshape(-1, 49) / 255


def load_data():
    images, labels, _ = load_images()
    return block_features(images), labels


def stratified_split(y, original_train_count, test_ratio=1 / 3, seed=42):
    """Özgün test örneklerinin tamamını testte tut ve hedef oranı tamamla."""
    rng = np.random.default_rng(seed)
    official_test = np.arange(original_train_count, len(y))
    train_parts = []
    test_parts = []
    for label in (0, 1):
        original_train_rows = np.flatnonzero(y[:original_train_count] == label)
        shuffled_rows = rng.permutation(original_train_rows)
        target_test_count = round(int(np.sum(y == label)) * test_ratio)
        official_test_count = int(np.sum(y[official_test] == label))
        extra_test_count = target_test_count - official_test_count
        test_parts.append(shuffled_rows[:extra_test_count])
        train_parts.append(shuffled_rows[extra_test_count:])

    test_parts.append(official_test)
    return np.concatenate(train_parts), np.concatenate(test_parts)


class StandardScaler:
    """Eğitim sütunlarından ortalama ve standart sapma öğrenir."""

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.zero_variance = self.std == 0
        self.std[self.zero_variance] = 1
        return self

    def transform(self, X):
        return (X - self.mean) / self.std
