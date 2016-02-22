import csv
import random

# noinspection PyPep8Naming
from Node import Node


# noinspection PyPep8Naming
class NeuralNetwork:
    def __init__(self):
        # The targets
        self.targets = dict()

        # The learning rate (ada)
        self.learningRate = 0.1

        # The list of layers (of nodes) in our network
        self.layers = list()

        # The output layer
        self.outputLayer = list()

        # The training and testing sets loaded from external data
        self.ratio = (2.0 / 3.0)
        self.trainingSet = list()
        self.testingSet = list()

    """
    Load an external dataset and convert the
    attribute values to floats
    """

    def loadDataset(self, filename):
        # Read all of the lines as text
        with open(filename, "rt") as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)

            # Randomize the order of the data
            random.shuffle(dataset)

            # Ensure that the training set is twice as big as the testing set
            for row in range(len(dataset)):
                if len(self.trainingSet) < len(dataset) * self.ratio:
                    self.trainingSet.append(dataset[row])
                else:
                    self.testingSet.append(dataset[row])
                for attribute in range(len(dataset[row]) - 1):
                    # Convert each data attribute into a float
                    dataset[row][attribute] = float(dataset[row][attribute])

    """
    Normalize our data such that for each instance,
    the sum of its attributes will always be 1.0
    """

    def normalize(self):
        normalizedTraining = list()
        normalizedTesting = list()

        # 1. Copy the training set into a temporary list
        #    (excluding the target identifier attribute)
        for i in range(len(self.trainingSet)):
            normalizedTraining.append(self.trainingSet[i][:-1])

        # 2. Normalize the data in the temporary list
        for i in range(len(normalizedTraining)):
            normalizedTraining[i] = [j / sum(normalizedTraining[i]) for j in normalizedTraining[i]]

        # 3. Replace the raw data with the normalized data
        for i in range(len(self.trainingSet)):
            self.trainingSet[i][:-1] = normalizedTraining[i]

        """Do the same for the testing set"""
        # 1. Copy the testing set into a temporary list
        #    (excluding the target identifier attribute)
        for i in range(len(self.testingSet)):
            normalizedTesting.append(self.testingSet[i][:-1])

        # 2. Normalize the data in the temporary list
        for i in range(len(normalizedTesting)):
            normalizedTesting[i] = [j / sum(normalizedTesting[i]) for j in normalizedTesting[i]]

        # 3. Replace the raw data with the normalized data
        for i in range(len(self.testingSet)):
            self.testingSet[i][:-1] = normalizedTesting[i]

    """
    Specify the size of the layers
    as well as the number of nodes in each layer
    """

    def createNetwork(self, hiddenLayers, targets):
        # Create the list of hidden layers
        for i in range(len(hiddenLayers)):
            # This list represents a new layer
            layer = list()
            for j in range(hiddenLayers[i]):
                node = Node()
                layer.append(node)
            self.layers.append(layer)

        # Create the output layer
        layer = list()
        for i in range(len(targets)):
            node = Node()
            layer.append(node)
        self.layers.append(layer)

        # Keep track of the outputs
        # (we will need to know the target values later)
        for i in range(len(targets)):
            key = 'target'
            key += str(i)
            self.targets[key] = targets[i]

    """
    Starting with inputs from the specified data instance,
    calculate the inputs and outputs
    """

    def feed(self, instance):
        # Layers
        for i in range(len(self.layers)):
            # Because each layer will have a different set of outputs
            outputs = list()

            # Nodes
            for j in range(len(self.layers[i])):
                if i == 0:
                    inputs = instance[:-1]

                self.layers[i][j].setInputs(inputs)
                self.layers[i][j].generateWeights()

                output = self.layers[i][j].getOutput()
                outputs.append(output)

            # Generate the inputs for the next layer
            inputs = list()
            for j in range(len(outputs)):
                inputs.append(outputs[j])

            # Save the final outputs of the last layer
            self.outputLayer = outputs

    def getClassification(self):
        # Map each output to its respective target value
        for i in range(len(self.targets)):
            targetKey = 'target'
            targetKey += str(i)

            if self.outputLayer[i] == max(self.outputLayer):
                return self.targets[targetKey]

    """
    Starting with inputs from the specified data instance,
    calculate the inputs and outputs
    """

    def feed(self, instance):
        # Layers
        for i in range(len(self.layers)):
            # Because each layer will have a different set of outputs
            outputs = list()
            # print()
            # print("Layer", i + 1)
            # print("--------------------")

            # Nodes
            for j in range(len(self.layers[i])):
                # print()
                if i == 0:
                    inputs = instance[:-1]

                # print("Node inputs:", inputs)
                self.layers[i][j].setInputs(inputs)
                self.layers[i][j].generateWeights()
                output = self.layers[i][j].getOutput()
                outputs.append(output)

                # Map the target value to the target type
                target = self.layers[i][j].target = instance[-1]
                if target == 'Iris-setosa':
                    self.layers[i][j].targetValue = 0
                elif target == 'Iris-virginica':
                    self.layers[i][j].targetValue = 1
                else:
                    self.layers[i][j].targetValue = 2

            # Generate the inputs for the next layer
            inputs = list()
            for j in range(len(outputs)):
                inputs.append(outputs[j])

            # Save the final outputs of the last layer
            self.outputLayer = outputs

    '''
    Starting in the output layer, find the error for each node
    '''
    def getOutputNodeError(self, activationJ, targetJ):
        # Calculate error for an output node
        return activationJ * ((1 - activationJ) * (activationJ - targetJ))

    '''
    Starting in the output layer, find the error for each node
    '''
    def getHiddenNodeError(self, activationJ, weightsJK, errorsK):
        # First, get the sum of the products
        sumOfProducts = 0

        # Loop through and get each weight multiplied by the error
        # of the node on the right
        for i in range(len(weightsJK)):
            sumOfProducts += ((weightsJK[i]) * (errorsK[i]))

        return activationJ * ((1 - activationJ) * sumOfProducts)

    '''
    Starting in the output layer, find the error for each node
    '''
    def calculateError(self):
        # Error calculation is different for output layer than it is
        # for hidden layer

        # Keep track of each node's error value. Structure it in the same
        # way that the network was built, so as to map each node's error to
        # the node

        errorList = list()

        layers = self.layers[::-1]

        for i in range(len(layers)):
            currentLayer = list()

            for j in range(len(layers[i])):
                if i == 0:
                    error = self.getOutputNodeError(layers[i][j].getOutput(), layers[i][j].targetValue)
                    currentLayer.append(error)
                else:
                    index = i - 1
                    error = self.getHiddenNodeError(layers[i][j].getOutput(), layers[index][j].weights, errorList[index])
                    currentLayer.append(error)

            errorList.append(currentLayer)

        return errorList

    '''
    Update each node's weights
    '''
    def updateWeights(self, errorList):
        # w = w - n(dj)(ai)
        oldWeights = list()
        newWeights = list()

        # Store the values of the original weights
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                for k in range(len(self.layers[i][j].weights)):
                    oldWeights.append(self.layers[i][j].weights[k])

        # Store the values of the new weights
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                for k in range(len(self.layers[i][j].weights)):
                    weight = self.layers[i][j].weights[k] - (self.learningRate * errorList[i][j] * self.layers[i][j].getOutput())
                    newWeights.append(weight)

        # Compare for debugging
        for i in range(len(oldWeights)):
            print("old: ", oldWeights[i], "new: ", newWeights[i])

        return newWeights

    """
    First, get the errors of all of the nodes
    then, update the weights accordingly
    """
    def propagateBack(self):
        # Get all of the errors in the network
        errorList = self.calculateError()
        errorList = errorList[::-1]

        # Update the weights
        self.updateWeights(errorList)




