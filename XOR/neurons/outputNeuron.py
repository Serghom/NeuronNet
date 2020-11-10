class OutputNeuron:
    def __init__(self):
        self.i = 0
        self.o = 0
        self.delta = 0

    def sigmoid(self):
        from math import exp
        x = self.i
        x = 1 / (1 + exp(-x))
        self.o = x

    def deltaBackRotation(self, ideal):
        # deltO = (OUTidel - OUTactual) * f'(IN)
        # f'(IN) = (1 - OUT) * OUT
        self.delta = (ideal - self.o) * ((1 - self.o) * self.o)
