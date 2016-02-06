from Node import Node
from NeuralNetwork import NeuralNetwork

""" IRIS """
nn = NeuralNetwork()
nn.loadDataset('datasets/iris.data')
nn.normalize()

# Create a node for each instance in the Iris training set
for i in range(len(nn.trainingSet)):

    # Convert attributes to inputs
    inputs = list()
    for j in range(len(nn.trainingSet[i]) - 1):  # But ignore target names (i.e. the last attribute)
        inputs.append(nn.trainingSet[i][j])

    # Create the node
    node = Node()
    node.setInputs(inputs)
    node.generateWeights()
    nn.addNode(node)

# Display the info for each node in the network
for i in range(len(nn.nodes)):
    print()
    print("Node number", i + 1)
    print(nn.nodes[i].bias['input'], "w =", nn.nodes[i].bias['weight'])
    for j in range(len(nn.nodes[i].inputs)):
        print(nn.nodes[i].inputs[j], "w =", nn.nodes[i].weights[j])
    print("The output from this node is:", nn.nodes[i].getOutput())


""" DIABETES """
nn = NeuralNetwork()
nn.loadDataset('datasets/pima-indians-diabetes.data')
nn.normalize()

# Create a node for each instance in the Diabetes training set
for i in range(len(nn.trainingSet)):

    # Convert attributes to inputs
    inputs = list()
    for j in range(len(nn.trainingSet[i]) - 1):  # But ignore target names (i.e. the last attribute)
        inputs.append(nn.trainingSet[i][j])

    # Create the node
    node = Node()
    node.setInputs(inputs)
    node.generateWeights()
    nn.addNode(node)

# Display the info for each node in the network
for i in range(len(nn.nodes)):
    print()
    print("Node number", i + 1)
    print(nn.nodes[i].bias['input'], "w =", nn.nodes[i].bias['weight'])
    for j in range(len(nn.nodes[i].inputs)):
        print(nn.nodes[i].inputs[j], "w =", nn.nodes[i].weights[j])
    print("The output from this node is:", nn.nodes[i].getOutput())
