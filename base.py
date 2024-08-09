import math
import numpy as np


class Perceptron:
    """
    n: numero de neuronios na camada de processamento.\n
    D: Vetor de Rótulos de Saída.\n
    X: Matriz de entradas.\n
    max_it: maximo de iterações (épocas).\n
    fator_aprendizagem: Fator de aprendizagem para calculo do bias e atualização da matriz de pesos.\n
    """

    def __init__(self, max_it, fator_aprendizagem, X, D, n) -> None:
        self.n = n  # numero de neuronios na camada de processamento
        self.X = np.array(X)  # Matriz de entrada
        self.D = np.array(D)  # Rotulos
        self.W = [[0 for len in range(self.X.shape[0])] for len in range(n)]
        # Matriz de pesos | Numero de linhas
        self.b = [0 for len in range(n)]  # Bias
        self.t = 1  # Época
        self.Error = 1  # Error
        self.max_it = max_it + 1  # Maximo de Épocas
        self.fator_aprendizagem = fator_aprendizagem  # Fator de aprendizagem
        self.ve = [[] for len in range(n)]  # vetor de erros

    def degrauFunction(self, value, j):
        return 1 if value[j] >= 0 else 0

    def start_training(self):
        while self.t < self.max_it and self.Error > 0:
            self.Error = 0
            x = self.X
            for i in range(self.n):  # neuronio
                d = self.D[i]
                for j in range(self.X.shape[1]):  # Numero de colunas (entradas)

                    w = self.W[i]
                    b = self.b[i]

                    y = self.degrauFunction(
                        (w @ x) + b, j
                    )  # @ -> Operador de produto escalar entre matrizes

                    e = d[j] - y

                    self.W[i] = (
                        w + (self.fator_aprendizagem * e) * x.T[j]
                    )  # -> transposta
                    self.b[i] = b + self.fator_aprendizagem * e * 1  # -> X[0] = 1
                    self.Error = self.Error + math.pow(e, 2)  # sum(math.pow(e, 2))

                    self.ve[i].append(e)
                    print(w)
                    print(self.b)

            self.t = self.t + 1

    def start_test(self, X, D):
        acertos = 0
        taxa_acertos = 0
        ve_teste = [[] for len in range(self.n)]

        for i in range(self.n):  # neuronio
            d = D[i]
            for j in range(self.X.shape[1]):
                y = self.degrauFunction((np.dot(self.W[i], X)) + self.b[i], j)
                e = d[j] - y
                ve_teste[i].append(e)

                if e == 0:
                    acertos = acertos + 1
        taxa_acertos = (acertos / self.X.shape[1]) * 100
        print(f"Acuracia: {taxa_acertos}%")


p = Perceptron(4, 0.1, [[0, 0, 1, 1], [0, 1, 0, 1]], [[0, 1, 1, 1]], 1)
p.start_training()
p.start_test([[0, 0, 1, 1], [1, 0, 1, 0]], [[1, 0, 1, 1]])
