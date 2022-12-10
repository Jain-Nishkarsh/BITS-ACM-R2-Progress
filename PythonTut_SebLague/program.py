import neuralnetwork as nn
import numpy as np

with np.load('E:\VS PYTHON\MachineLearning\mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    
layer_sizes = (784,5,10)

net = nn.NeuralNetwork(layer_sizes)
prediction = net.predict(training_images)
print(np.argmax(prediction[0]))
