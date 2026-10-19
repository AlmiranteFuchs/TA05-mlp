"""MLP from scratch (numpy) - versao corrigida da implementacao de referencia.

Mesma arquitetura do notebook da disciplina (1 camada escondida, one-hot,
SGD por amostra, MSE), com os bugs corrigidos e instrumentacao de
convergencia (loss/acuracia por epoca, treino e validacao).

Bugs do original que esta versao corrige:
  - predict() nao aplicava a funcao de ativacao (forward "linear")
  - BIAS_output era atualizado dentro do loop dos neuronios escondidos
    (bias de saida andava hiddenLayer vezes mais rapido)
  - erro de epoca acumulado dentro do loop das saidas e com fator 0.05
    (valor reportado ~3x maior e escala errada)
  - params=None quebrava: 'iasHiddenValue' (typo) nunca virava BiasHiddenValue
  - classes_number fixo em 3
  - sem seed / sem conjunto de validacao / sem historico por epoca
"""
import numpy as np


class MultiLayerPerceptron:
    """4 -> H -> 3 (ou qualquer n_in/n_hidden/n_out) com backprop por amostra."""

    ativacao = {
        'sigmoid': lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))),
        'tanh':    lambda x: np.tanh(x),
        'relu':    lambda x: np.maximum(x, 0.0),
    }
    derivada = {  # derivadas escritas em funcao da SAIDA da ativacao (como no original)
        'sigmoid': lambda y: y * (1.0 - y),
        'tanh':    lambda y: 1.0 - y ** 2,
        'relu':    lambda y: 1.0 * (y > 0),
    }

    def __init__(self, params=None, seed=None):
        p = params or {}
        self.inputLayer   = p.get('InputLayer', 4)
        self.hiddenLayer  = p.get('HiddenLayer', 5)
        self.outputLayer  = p.get('OutputLayer', 3)
        self.learningRate = p.get('LearningRate', 0.005)
        self.max_epochs   = p.get('Epocas', 600)
        act = p.get('ActivationFunction', 'sigmoid')
        self.activation   = self.ativacao[act]
        self.deriv        = self.derivada[act]
        self.act_name     = act
        self.params       = params

        rng = np.random.default_rng(seed)
        # mesmo intervalo do original: uniforme [-1, +1]
        self.W_hidden = rng.uniform(-1, 1, (self.inputLayer, self.hiddenLayer))
        self.W_output = rng.uniform(-1, 1, (self.hiddenLayer, self.outputLayer))
        self.b_hidden = np.full(self.hiddenLayer, p.get('BiasHiddenValue', -1.0),
                                dtype=float)
        self.b_output = np.full(self.outputLayer, p.get('BiasOutputValue', -1.0),
                                dtype=float)

    # ---------------- forward ----------------
    def _forward(self, x):
        """Retorna (saida_escondida, saida_rede) ja com ativacao aplicada."""
        h = self.activation(x @ self.W_hidden + self.b_hidden)
        o = self.activation(h @ self.W_output + self.b_output)
        return h, o

    def predict_proba(self, X):
        """Saidas ativadas da rede para cada linha de X."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self._forward(x)[1] for x in X])

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    # ---------------- loss ----------------
    def mse(self, X, y):
        """MSE sobre os neuronios de saida (one-hot), media por amostra."""
        Y = np.eye(self.outputLayer)[np.asarray(y, dtype=int)]
        return float(np.mean((Y - self.predict_proba(X)) ** 2))

    # ---------------- training ----------------
    def _backprop(self, x, target):
        h, o = self._forward(x)
        delta_o = (o - target) * self.deriv(o)                  # dL/dz_out
        delta_h = (self.W_output @ delta_o) * self.deriv(h)     # dL/dz_hidden (usa W antigo)
        self.W_output -= self.learningRate * np.outer(h, delta_o)
        self.b_output -= self.learningRate * delta_o
        self.W_hidden -= self.learningRate * np.outer(x, delta_h)
        self.b_hidden -= self.learningRate * delta_h

    def fit(self, X, y, X_val=None, y_val=None, log_every=10, verbose=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        hist = {'epoch': [], 'train_loss': [], 'train_acc': [],
                'val_loss': [], 'val_acc': []}
        for epoch in range(1, self.max_epochs + 1):
            for xi, yi in zip(X, y):
                self._backprop(xi, np.eye(self.outputLayer)[yi])
            if epoch == 1 or epoch % log_every == 0 or epoch == self.max_epochs:
                hist['epoch'].append(epoch)
                hist['train_loss'].append(self.mse(X, y))
                hist['train_acc'].append(self.score(X, y))
                if X_val is not None:
                    hist['val_loss'].append(self.mse(X_val, y_val))
                    hist['val_acc'].append(self.score(X_val, y_val))
                if verbose:
                    print(f"Epoch {epoch:4d} - MSE: {hist['train_loss'][-1]:.4f}")
        return hist
