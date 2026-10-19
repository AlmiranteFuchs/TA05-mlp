"""Port fiel do MLP do notebook de referencia (com os bugs preservados).

Serve so para comparacao: mede o "erro" do jeito que o original mede
(e reporta errado) e tambem as metricas verdadeiras por epoca.
Diferencas em relacao ao notebook: sem plots/pandas dentro do fit e com
seed controlada para reproducibilidade - o resto e identico, bugs inclusos.
"""
import numpy as np
import pandas as pd


class MLPReferencia:
    def __init__(self, params=None, seed=None):
        self.params = params
        if params is None:
            self.inputLayer = 4
            self.hiddenLayer = 5
            self.outputLayer = 3
            self.learningRate = 0.005
            self.max_epochs = 600
            self.iasHiddenValue = -1          # typo do original: params=None quebra abaixo
            self.BiasOutputValue = -1
            self.activation = self.ativacao['sigmoid']
            self.deriv = self.derivada['sigmoid']
        else:
            self.inputLayer = params['InputLayer']
            self.hiddenLayer = params['HiddenLayer']
            self.outputLayer = params['OutputLayer']
            self.learningRate = params['LearningRate']
            self.max_epochs = params['Epocas']
            self.BiasHiddenValue = params['BiasHiddenValue']
            self.BiasOutputValue = params['BiasOutputValue']
            self.activation = self.ativacao[params['ActivationFunction']]
            self.deriv = self.derivada[params['ActivationFunction']]

        rng = np.random.default_rng(seed)
        self.WEIGHT_hidden = rng.uniform(-1, 1, (self.inputLayer, self.hiddenLayer)).tolist()
        self.WEIGHT_output = rng.uniform(-1, 1, (self.hiddenLayer, self.outputLayer)).tolist()

        # com params=None isto estoura AttributeError (iasHiddenValue != BiasHiddenValue)
        # BUG sutil: array de int; BIAS[j] -= 0.x trunca p/ int -> bias nunca aprende
        self.BIAS_hidden = np.array([self.BiasHiddenValue for i in range(self.hiddenLayer)])
        self.BIAS_output = np.array([self.BiasOutputValue for i in range(self.outputLayer)])
        self.classes_number = 3

    ativacao = {
        'sigmoid': (lambda x: 1 / (1 + np.exp(-x))),
        'tanh':    (lambda x: np.tanh(x)),
        'Relu':    (lambda x: x * (x > 0)),
    }
    derivada = {
        'sigmoid': (lambda x: x * (1 - x)),
        'tanh':    (lambda x: 1 - x ** 2),
        'Relu':    (lambda x: 1 * (x > 0)),
    }

    def Backpropagation_Algorithm(self, x):
        DELTA_output = []
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = ((-1) * (ERROR_output) * self.deriv(self.OUTPUT_L2))

        # BUG: BIAS_output atualizado dentro do loop i (anda hiddenLayer vezes por amostra)
        for i in range(self.hiddenLayer):
            for j in range(self.outputLayer):
                self.WEIGHT_output[i][j] -= (self.learningRate * (DELTA_output[j] * self.OUTPUT_L1[i]))
                self.BIAS_output[j] -= (self.learningRate * DELTA_output[j])

        # quirk: usa WEIGHT_output ja atualizado para propagar o erro
        delta_hidden = np.matmul(self.WEIGHT_output, DELTA_output) * self.deriv(self.OUTPUT_L1)

        for i in range(self.inputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= (self.learningRate * (delta_hidden[j] * x[i]))
                self.BIAS_hidden[j] -= (self.learningRate * delta_hidden[j])

    def predict(self, X, y):
        """BUG: forward sem funcao de ativacao (argmax nao muda, mas as 'saidas' sim)."""
        my_predictions = []
        forward = np.matmul(X, self.WEIGHT_hidden) + self.BIAS_hidden
        forward = np.matmul(forward, self.WEIGHT_output) + self.BIAS_output
        for i in forward:
            my_predictions.append(max(enumerate(i), key=lambda x: x[1])[0])

        array_score = []
        names = ['Iris-setosa', 'Iris-versicolour', 'Iris-virginica']
        for i in range(len(my_predictions)):
            array_score.append([i, names[my_predictions[i]], my_predictions[i], y[i]])
        dataframe = pd.DataFrame(array_score, columns=['_id', 'class', 'output', 'hoped_output'])
        return my_predictions, dataframe

    def score(self, X, y):
        pred, _ = self.predict(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
        return float(np.mean(np.array(pred) == np.asarray(y, dtype=int)))

    def predict_activated(self, X):
        """O que o predict DEVERIA fazer: forward com ativacao + argmax."""
        X = np.asarray(X, dtype=float)
        h = self.activation(X @ np.array(self.WEIGHT_hidden) + self.BIAS_hidden)
        o = self.activation(h @ np.array(self.WEIGHT_output) + self.BIAS_output)
        return np.argmax(o, axis=1)

    def fit(self, X, y, X_val=None, y_val=None, log_every=50):
        """Mesma logica do original; retorna historico com erro-reportado e metricas reais."""
        count_epoch = 1
        total_error = 0
        n = len(X)
        hist = {'epoch': [], 'reported_error': [], 'train_loss': [],
                'train_acc': [], 'val_loss': [], 'val_acc': []}
        while count_epoch <= self.max_epochs:
            for idx, inputs in enumerate(X):
                self.output = np.zeros(self.classes_number)
                self.OUTPUT_L1 = self.activation(np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden)
                self.OUTPUT_L2 = self.activation(np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output)
                if y[idx] == 0:
                    self.output = np.array([1, 0, 0])
                elif y[idx] == 1:
                    self.output = np.array([0, 1, 0])
                elif y[idx] == 2:
                    self.output = np.array([0, 0, 1])

                square_error = 0
                # BUG: total_error acumulado dentro do loop das saidas
                # (soma parciais: pesa erro[0] 3x, erro[1] 2x, erro[2] 1x) e fator 0.05
                for i in range(self.outputLayer):
                    erro = (self.output[i] - self.OUTPUT_L2[i]) ** 2
                    square_error = square_error + (0.05 * erro)
                    total_error = total_error + square_error
                self.Backpropagation_Algorithm(inputs)

            total_error = total_error / n
            if count_epoch % log_every == 0 or count_epoch == 1:
                hist['epoch'].append(count_epoch)
                hist['reported_error'].append(total_error)
                hist['train_loss'].append(self._true_mse(X, y))
                hist['train_acc'].append(self.score(X, y))
                if X_val is not None:
                    hist['val_loss'].append(self._true_mse(X_val, y_val))
                    hist['val_acc'].append(self.score(X_val, y_val))
            total_error = 0
            count_epoch += 1
        return hist

    def _true_mse(self, X, y):
        X = np.asarray(X, dtype=float)
        Y = np.eye(3)[np.asarray(y, dtype=int)]
        preds = []
        for x in X:
            o1 = self.activation(x @ np.array(self.WEIGHT_hidden) + self.BIAS_hidden)
            o2 = self.activation(o1 @ np.array(self.WEIGHT_output) + self.BIAS_output)
            preds.append(o2)
        return float(np.mean((Y - np.array(preds)) ** 2))
