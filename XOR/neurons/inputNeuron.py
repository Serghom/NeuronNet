class InputNeuron:
    def __init__(self, i=0):
        self.i = i
        self.o = i

    def setValue(self, i):
        self.o = self.i = i