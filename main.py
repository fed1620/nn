from NeuralNetwork import NeuralNetwork

""" Which dataset would you like to test?"""
# 0 = Iris dataset
# 1 = Pima Indians dataset
dataset = 0

if dataset == 0:
    filename = "datasets/iris.data"
    targets = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
elif dataset == 1:
    filename = "datasets/pima-indians-diabetes.data"
    targets = [0, 1]

# Create the network
nn = NeuralNetwork()
nn.loadDataset(filename)
nn.normalize()
nn.createNetwork([2, 3], targets)
numCorrect = 0

# Make the predictions
for i in range(len(nn.testingSet)):
    nn.feed(nn.testingSet[i])
    print("Instance", i + 1, ": predicted =", nn.getClassification(), "actual =", nn.testingSet[i][-1])
    if nn.getClassification() == nn.testingSet[i][-1]:
        numCorrect += 1

# Output the accuracy
accuracy = float(numCorrect) / float(len(nn.testingSet)) * 100
print("Accuracy:", float("{0:.1f}".format(accuracy)), '%')