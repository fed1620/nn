import random
from scipy.special import expit


# noinspection PyPep8Naming
class Node:
    def __init__(self):
        self.threshold = 0
        self.bias = dict()
        self.inputs = list()
        self.weights = list()

    """
    Set the inputs for this node
    """
    def setInputs(self, inputs):
        for i in range(len(inputs)):
            self.inputs.append(inputs[i])

    """
    Randomly generate different weights for
    each input
    """
    def generateWeights(self):
        # Handle the bias input
        self.bias['input'] = -1.0
        self.bias['weight'] = float("{0:.1f}".format(random.uniform(-1.0, 1.0)))

        # And then the rest of the inputs
        for i in range(len(self.inputs)):
            self.weights.append(float("{0:.1f}".format(random.uniform(-1.0, 1.0))))

    """
    Sum of the products will either be above
    the threshold, or beneath it
    """
    def getOutput(self):
        # Start with the bias node
        sum = self.bias['input'] * self.bias['weight']

        # And then add the rest of the inputs * weights
        for i in range(len(self.inputs)):
            sum = sum + self.inputs[i] * self.weights[i]

        return expit(sum)
