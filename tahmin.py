import argparse
import json
from pathlib import Path

import numpy as np

from modeller import sigmoid
from veri_isleme import PROJECT_ROOT, block_features, load_images, stratified_split


DEFAULT_MODEL = PROJECT_ROOT / 'sonuclar' / 'model_parametreleri.json'


def predict(features, model_path=None):
    """49 blok ortalamasını kaydedilen değerlerle ölçekle ve iki skoru hesapla."""
    path = Path(model_path) if model_path else DEFAULT_MODEL
    bundle = json.loads(path.read_text(encoding='utf-8'))
    preprocessing = bundle['preprocessing']
    blocks = np.asarray(features, dtype=float)
    feature_count = len(preprocessing['feature_order'])
    if blocks.shape != (feature_count,):
        raise ValueError(f'0-1 aralığında {feature_count} blok ortalaması gerekli.')

    train_mean = preprocessing['mean_train']
    train_std = preprocessing['std_train_ddof0']
    scaled = (blocks - train_mean) / train_std
    threshold = bundle['threshold']
    result = {'threshold': threshold}
    for name, details in bundle['models'].items():
        weights = details['weights_standardized']
        bias = details['bias_standardized']
        score = float(scaled @ weights + bias)
        if name != 'dogrusal':
            score = float(sigmoid(score))
        result[name] = {'score': score, 'class': int(score >= threshold)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('features', type=float, nargs='*', metavar='X')
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument('--row', type=int, help='0/1 alt kümesinde 1 tabanlı kaynak satırı')
    choices.add_argument('--test-sira', type=int, help='Test kümesinde 0 tabanlı örnek sırası')
    choices.add_argument('--gorsel', type=Path, help='28x28, 0-255 değerli gri ton .npy dosyası')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    if args.features and any(value is not None for value in (args.row, args.test_sira, args.gorsel)):
        parser.error('Özellikler ya da tek bir örnek seçeneği girilmeli.')
    actual = None
    if args.gorsel is not None:
        image = np.load(args.gorsel, allow_pickle=False)
        if image.shape != (28, 28):
            parser.error('28x28 boyutunda bir .npy dizisi gerekli.')
        features = block_features(image)[0]
    elif args.features:
        features = args.features
    else:
        images, labels, original_train_count = load_images()
        all_features = block_features(images)
        row = args.row
        if row is None:
            _, test_idx = stratified_split(labels, original_train_count)
            index = 0 if args.test_sira is None else args.test_sira
            if not 0 <= index < len(test_idx):
                parser.error(f'Test sırası 0-{len(test_idx) - 1} aralığında olmalı.')
            row = int(test_idx[index]) + 1

        if not 1 <= row <= len(labels):
            parser.error(f'Satır 1-{len(labels)} aralığında olmalı.')
        features = all_features[row - 1]
        actual = int(labels[row - 1])

    result = predict(features, args.model)
    if actual is not None:
        result['gercek_rakam'] = actual
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
