from dataclasses import dataclass

import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


@dataclass
class EgitimAyarlari:
    learning_rate: float = 0.5
    max_iter: int = 200000
    tolerance: float = 1e-4
    log_every: int = 100

    def validate(self):
        if self.learning_rate <= 0:
            raise ValueError("ogrenme orani hatasi")
        if self.max_iter < 1 or self.log_every < 1:
            raise ValueError("iterasyon hatasi")
        if self.tolerance <= 0:
            raise ValueError("tolerans hatasi")


class TemelRegresyon:
    name = "temel"

    def __init__(self, settings=None):
        self.settings = settings or EgitimAyarlari()
        self.settings.validate()
        self.w = None
        self.b = 0.0
        self.history = []

    def loss_gradient(self, X, y, w, b):
        raise NotImplementedError

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.history = []
        self.stop_reason = "maksimum_iterasyon"
        settings = self.settings

        for iteration in range(settings.max_iter + 1):
            loss, dw, db = self.loss_gradient(X, y, self.w, self.b)
            gradient_norm = float(np.sqrt(dw @ dw + db * db))
            if np.isnan(loss) or np.isnan(gradient_norm):
                raise ValueError("Egitim iraksadi ogrenme oranini kucultun")
            converged = gradient_norm <= settings.tolerance
            is_last_step = converged or iteration == settings.max_iter
            if iteration < 100 or iteration % settings.log_every == 0 or is_last_step:
                self.history.append((iteration, float(loss), gradient_norm))
            if is_last_step:
                self.n_iter = iteration
                self.final_loss = float(loss)
                self.gradient_norm = gradient_norm
                if converged:
                    self.stop_reason = "gradyan_toleransi"
                break

            self.w = self.w - settings.learning_rate * dw
            self.b = self.b - settings.learning_rate * db
        return self

    def decision_function(self, X):
        if self.w is None:
            raise RuntimeError("fit cagrilmali")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(self.w):
            raise ValueError("tahmin verisinin ozellik sayisi uyusmuyor")
        return X @ self.w + self.b

    def predict_score(self, X):
        raise NotImplementedError

    def predict(self, X, threshold=0.5):
        if not (0 < threshold < 1):
            raise ValueError("Karar esigi 0 ile 1 arasinda olmali")
        return (self.predict_score(X) >= threshold).astype(int)


class DogrusalRegresyon(TemelRegresyon):
    name = "dogrusal"

    def loss_gradient(self, X, y, w, b):
        prediction = X @ w + b
        error = prediction - y
        sample_count = len(y)
        loss = float(error @ error / (2 * sample_count))
        dw = X.T @ error / sample_count
        db = float(error.mean())
        return loss, dw, db

    def predict_score(self, X):
        return self.decision_function(X)


class LojistikRegresyon(TemelRegresyon):
    name = "lojistik"

    def loss_gradient(self, X, y, w, b):
        z = X @ w + b
        probability = sigmoid(z)
        error = probability - y
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        dw = X.T @ error / len(y)
        db = float(error.mean())
        return loss, dw, db

    def predict_score(self, X):
        return sigmoid(self.decision_function(X))
