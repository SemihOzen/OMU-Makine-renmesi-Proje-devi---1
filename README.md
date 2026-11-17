# MNIST 0 ve 1 üzerinde iki regresyonun karşılaştırılması

Proje, el yazısıyla yazılmış bir görüntüyü `0` veya `1` olarak sınıflandırır. Doğrusal ve lojistik regresyon aynı test görüntülerinde karşılaştırılır. İki modelin tahmini, maliyet fonksiyonu, gradyanı ve Batch Gradient Descent adımları `modeller.py` içinde NumPy ile yazılmıştır.

Proje raporu: [Düzenlenebilir Word](rapor/MNIST_Proje_Raporu.docx) · [Teslim PDF’si](rapor/MNIST_Proje_Raporu.pdf)

## Veri

MNIST veri kümesinden ikili sınıflandırma görevi için `0` ve `1` sınıflarına ait **14.780 görüntünün tamamı** (28x28 gri tonlu) kullanılmıştır. Kaynak görüntüler ve etiketler `veri/ham/mnist.npz` içindedir.

| Küme | 0 | 1 | Toplam |
|---|---:|---:|---:|
| Eğitim | 4.602 | 5.251 | **9.853** |
| Test | 2.301 | 2.626 | **4.927** |
| Toplam | 6.903 | 7.877 | **14.780** |

Veri kümesi yaklaşık **%66,67 eğitim ve %33,33 test** oranıyla bölünmüştür. Özgün MNIST testindeki 2.115 adet 0/1 görüntünün tamamı test kümesinde tutulmuş, kalan test örnekleri özgün eğitim kümesinden sınıf oranları korunarak (`seed=42`) seçilmiştir.

Her 28x28 görüntü 4x4 piksel bloklarına ayrılır. Her blok için 16 pikselin ortalaması alınır ve 255'e bölünerek 7x7 düzeninde **49 özellik** çıkarılır. Ardından her özellik sütunu eğitim kümesinin ortalaması ve standart sapması ile standartlaştırılır.

## Çalıştırma

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python proje.py
python tahmin.py --test-sira 0
```

Eğitim komutu çıktıları `sonuclar/` altında üretir. Kaydedilmiş modelle tahmin denemek için `python tahmin.py --test-sira 0` veya `python tahmin.py --row 5396` kullanılabilir. Kendi 28x28 `.npy` görseliniz için `python tahmin.py --gorsel rakam.npy` komutunu çalıştırabilirsiniz.

## Modeller ve karşılaştırma

Doğrusal model `z = X @ w + b` çıktısını, lojistik model `sigmoid(z)` çıktısını kullanır. İkisinde de skor 0,5 veya üstündeyse sınıf 1 seçilir. Doğrusal model yarım ortalama kare hatayı, lojistik model ikili çapraz entropiyi azaltır. Her güncellemede iki gradyan da aynı eski parametrelerden hesaplanır.

Ortak öğrenme oranı, yalnız eğitim matrisiyle hesaplanan kare hata Hessian'ının en büyük özdeğerinin tersidir. Bu deneyde `α ≈ 0,09582658`. Doğrusal model 770 adımda gradyan toleransına ulaştı. Lojistik model 50.000 adımlık üst sınırda durdu.

| Model | Doğruluk | 0 yakalama | 1 yakalama | Hata | Özgün 2.115 testte doğruluk |
|---|---:|---:|---:|---:|---:|
| Doğrusal regresyon | %99,80 | %99,70 | %99,89 | 10 | %99,81 |
| Lojistik regresyon | %99,84 | %99,70 | %99,96 | 8 | %99,81 |

Doğrusal modelin `(TN, FP, FN, TP)` sayıları `(2294, 7, 3, 2623)`. Lojistik modelinkiler `(2294, 7, 1, 2625)`. Eğitimde çoğunluk olan 1'i her görüntü için seçmek bu testte %53,30 doğruluk verir.

## Dosyalar

| Dosya veya klasör | İçerik |
|---|---|
| `veri_isleme.py` | Kaynak dosyayı yükleme, 49 özellik, bölme ve standardizasyon |
| `modeller.py` | İki kayıp, gradyanlar, Batch GD ve tahmin |
| `metrikler.py` | Hata matrisi, doğruluk, precision, recall, F1 ve ROC-AUC |
| `proje.py` | İki modeli eğitme ve ortak testte karşılaştırma |
| `tahmin.py` | Kaydedilen model parametreleriyle tek örnek tahmini |
| `sonuclar/ozet.json` | Ayarlar, bölünme, eğitim durumu ve metrikler |
| `sonuclar/model_parametreleri.json` | Kaydedilen iki model ve eğitimden öğrenilen ölçek değerleri |
| `sonuclar/test_tahminleri.csv` | 4.927 test görüntüsünün etiketleri ve tahminleri |
| `sonuclar/grafikler/` | Örnek görüntüler, eğitim kaybı, hata matrisleri, metrikler, ROC ve skor dağılımları |
| `rapor/` | Düzenlenebilir Word raporu ve teslim PDF'si |

Çıktılarda `original_row` 0/1 alt kümesindeki 1 tabanlı satırdır. İlk 12.665 satır özgün MNIST eğitim dosyasından, son 2.115 satır özgün test dosyasından gelir. `metrikler.csv` içindeki `ozgun_test`, 4.927 test görüntüsünün içinde yer alan bu 2.115 görüntüye ait ayrı ölçümdür.

Kaynak atfı Yann LeCun ve Corinna Cortes, *MNIST Database of Handwritten Digits*, NIST el yazısı verisinden türetilmiştir. [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) · [Yayımlanan dosya ve sağlama toplamı](https://raw.githubusercontent.com/keras-team/keras/master/keras/src/datasets/mnist.py)
