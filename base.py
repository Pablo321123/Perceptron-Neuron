import math
import numpy as np
import pandas as pd


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
        self.ve = [[] for len in range(self.X.shape[1])]  # vetor de erros

        self.sigmoid = lambda x, j: 1 / (np.exp(-x[j]))
        self.degrau = lambda value, j: 1 if value[j] >= 0 else 0

    def start_training(self, sigmoid=False):
        while self.t < self.max_it and self.Error > 0:
            self.Error = 0
            x = self.X
            for j in range(self.X.shape[1]):  # Numero de colunas (entradas)
                Y = []  # [[] for len in range(self.n)]
                d = self.D[j]
                for i in range(self.n):  # neuronio
                    Y.append([])  # Cria a resposta do próximo neuronio
                    w = self.W[i]
                    b = self.b[i]

                    # @ -> Operador de produto escalar entre matrizes
                    y_i = (
                        self.degrau((w @ x) + b, j)
                        if not sigmoid
                        else self.sigmoid((w @ x) + b, j)
                    )
                    Y[i] = y_i  # Vetor de saida de todos os neuronios

                if sigmoid:
                    max_y = max(Y)
                    Y = [1 if x == max_y else 0 for x in Y]

                # Fazer as verificações para verificar se as colunas são iguais
                e = d - Y
                for i in range(self.n):
                    x_t = x.T[j]
                    self.W[i] = (
                        self.W[i] + (self.fator_aprendizagem * e[i]) * x_t
                    )  # -> transposta
                    self.b[i] = self.b[i] + (
                        self.fator_aprendizagem * e[i] * 1
                    )  # -> X[0] = 1
                    self.Error = self.Error + math.pow(e[i], 2)  # sum(math.pow(e, 2))
                self.ve[j].append(e)
                # print(Y)

            self.t = self.t + 1

        print("Pesos:\n")
        print(pd.DataFrame(self.W))
        print("\nBias:\n")
        print(pd.DataFrame(self.b))

    def start_test(self, X_teste, D_teste, sigmoid=False):
        acertos = 0
        taxa_acertos = 0
        ve_teste = [[] for len in range(self.n)]
        X = np.array(X_teste)

        for j in range(X.shape[1]):
            Y = []
            d = np.array(D_teste[j])
            for i in range(self.n):  # neuronio
                Y.append([])
                y_i = (
                    self.degrau((self.W[i] @ X) + self.b[i], j)
                    if not sigmoid
                    else self.sigmoid((self.W[i] @ X) + self.b[i], j)
                )  # W[i] peso já treinado do neuronio i
                Y[i] = y_i

            if sigmoid:
                max_y = max(Y)
                Y = [1 if x == max_y else 0 for x in Y]

            e = d - Y
            ve_teste[i].append(e)

            if max(e) == 0 and min(e) == 0:
                acertos = acertos + 1

        taxa_acertos = (acertos / X.shape[1]) * 100
        print(f"Acuracia: {taxa_acertos}%")
        
        return taxa_acertos


def loadData(fileName):
    glass_list = pd.read_csv(f"../Perceptron-Neuron/data/{fileName}")
    return glass_list


# p = Perceptron(4, 0.1, [[0, 0, 1, 1], [0, 1, 0, 1]], [[0, 1, 1, 1]], 1)
# p.start_training()
# p.start_test([[0, 0, 1, 1], [1, 0, 1, 0]], [[1, 0, 1, 1]])
