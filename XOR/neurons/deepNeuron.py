class DeepNeuron:
    def __init__(self):
        self.i = 0
        self.o = 0
        self.delta = 0

    def sigmoid(self):
        from math import exp
        x = self.i
        x = 1 / (1 + exp(-x))
        self.o = x

    def deltaBackRotation(self, w, delt):
        # deltH = f'(IN) * E(wi * deltO)
        # f'(IN) = (1 - OUT) * OUT
        self.delta = ((1 - self.o) * self.o) * w * delt