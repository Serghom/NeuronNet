from neurons.inputNeuron import InputNeuron
from neurons.deepNeuron import DeepNeuron
from neurons.outputNeuron import OutputNeuron
from numpy import mean

inputSet = [[0, 0], [1, 1], [1, 0], [0, 1]]
goodAnswerSet = [0, 0, 1, 1]

w = [[0.45, 0.78], [-0.12, 0.13], [1.5, -2.3]]

inputNeurons = (InputNeuron(), InputNeuron())
deepNeurons = (DeepNeuron(), DeepNeuron())
outputNeuron = OutputNeuron()

# скорость обучения
eps = 0.7
# момент
a = 0.1

for epoch in range(10000):
    answerError = []
    for trainSet in range(4):
        inputNeurons[0].setValue(inputSet[trainSet][0])
        inputNeurons[1].setValue(inputSet[trainSet][1])

        print('in: {}  {}'.format(inputNeurons[0].o, inputNeurons[1].o))

        # синапсы в глубокие нейроны
        for i in range(2):
            synapse1 = w[0][i] * inputNeurons[0].o
            synapse2 = w[1][i] * inputNeurons[1].o
            deepNeurons[i].i = synapse1 + synapse2
            deepNeurons[i].sigmoid()

            print('deep N{}: {} -> {}'.format(i, deepNeurons[i].i, deepNeurons[i].o))

        # синапсы на выводящий нейрон
        synapse = (w[2][0] * deepNeurons[0].o) + (w[2][1] * deepNeurons[1].o)
        outputNeuron.i = synapse
        outputNeuron.sigmoid()

        print('output: {} -> {}\n-=-=-=-=-=-=-=-=-'.format(round(outputNeuron.i, 2), round(outputNeuron.o, 2)))

        error = pow((goodAnswerSet[trainSet] - outputNeuron.o), 2)
        answerError.append(error)

        # обучение Метод обратного распространения
        # вычисляем дельту выходного нейрона
        outputNeuron.deltaBackRotation(goodAnswerSet[trainSet])
        # изменение весов синопсов к выходному нейрону
        for i in range(2):
            # вычисляем дельты глубоких нейронов
            deepNeurons[i].deltaBackRotation(w[2][i], outputNeuron.delta)
            # вычисляем градиет глубоких нейронов
            grad = outputNeuron.delta * deepNeurons[i].o
            # вычисляем новый вес синопса через формулу МОР:
            # deltWi = (eps * GRADw) + (a * deltw_i-1)
            # w = w + deltWi
            w[2][i] += eps * grad

        # изменение весов синопсов от входных нейронов к глубоким
        for i in range(2):
            for j in range(2):
                grad = deepNeurons[j].delta * inputNeurons[i].o
                w[i][j] += (eps * grad) + (a * (outputNeuron.delta * deepNeurons[j].o))

    print('=======================')
    print('epoch: {}'.format(epoch))
    print('error: {}%'.format(round(mean(answerError), 2)))
    print('weight: {}'.format(w))
    print('=-=-=-=-=-=-=-=-=-=-=-=')
