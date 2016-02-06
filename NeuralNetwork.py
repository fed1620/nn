import csv
import random


# noinspection PyPep8Naming
class NeuralNetwork:
    def __init__(self):
        # The nodes/neurons in our network
        self.nodes = list()

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
            normalizedTraining[i] = [j/sum(normalizedTraining[i]) for j in normalizedTraining[i]]

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
            normalizedTesting[i] = [j/sum(normalizedTesting[i]) for j in normalizedTesting[i]]

        # 3. Replace the raw data with the normalized data
        for i in range(len(self.testingSet)):
            self.testingSet[i][:-1] = normalizedTesting[i]

    """
    Add a node to our list of nodes
    """
    def addNode(self, node):
        self.nodes.append(node)
