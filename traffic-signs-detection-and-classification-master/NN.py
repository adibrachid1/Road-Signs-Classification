import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import csv
import cv2


class Dataset:
    def __init__(self, data_dir, labels_path, data='train', n_samples=None):
        self.X = []
        self.y = []
        self.index = 0
        data_dir = Path(data_dir)
        with open(labels_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                label = row[-2]
                img_name = Path(row[-1]).name
                if data == 'train':
                    img_path = data_dir/label/img_name
                else:
                    img_path = data_dir/img_name
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.X.append(img)
                self.y.append(np.eye(43)[int(label)])
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y).reshape(-1,43,1)
        print(self.X.shape, self.y.shape)
        if n_samples != None:
            random_indexes = np.random.choice(list(range(len(self.X))), n_samples)
            self.X = self.X[random_indexes]
            self.y = self.y[random_indexes]
        self.len = len(self.y)    # Number of samples in the dataset (accessible through len(dataset))
        self.indices = list(range(self.len)) # List of indices used to pick samples in a random order
    def __len__(self):
        return self.len
    
    def next_sample(self):
        if self.index == self.len: # If we arrived at the end of the dataset...
            self.index = 0 # then reset the pointer
            np.random.shuffle(self.indices) # and shuffle the dataset
        i = self.index
        i = self.indices[i]
        self.index += 1
        return (self.X[i], self.y[i])


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    res = s*(1-s)
    return res

def argmax(l):
    maxval = max(l)
    for i in range(len(l)):
        if l[i] == maxval:
            return i

class Neuron:
    def __init__(self, inputs):
        # inputs must be an np array!
        n_inputs = len(inputs)
        self.inputs = inputs
        self.weights = np.random.rand(n_inputs) # weights and bias
        self.bias = np.random.rand()
        self.weights = self.weights.reshape(-1,1)
        self.bias = np.asarray(self.bias).reshape(-1,1)
        self.a = 0. # Activation of the neuron
        self.z = .5 # Output of the neuron
        self.d_weights = np.asarray([0. for _ in range(n_inputs)]).reshape(-1,1) # Derivatives of the loss wrt weights
        self.d_a = 0. # Derivative of the activation
    def feedforward(self):
        input_layer_as_array = np.asarray([self.inputs[i] for i in range(len(self.inputs))]).reshape(-1,1)
        res = np.dot(input_layer_as_array.T, self.weights) + self.bias
        self.a = res
        self.z = sigmoid(res)

class Layer:
    def __init__(self, inlayer, n_neurons):
        self.len = n_neurons # Number of neurons in the layer (accessible through len(layer))
        self.neurons = [] # List of neurons of the layer
        self.input = inlayer # Previous layer (or a dataset)
        n_inputs = len(inlayer) # Number of inputs
        for _ in range(n_neurons):
            self.neurons.append(Neuron(inlayer)) # Initialize neurons

    def __len__(self):
        return self.len

    def __getitem__(self, key):
        item = np.asarray(self.neurons[key].z).reshape(-1,1)
        return item

    def feedforward(self):
        for n in self.neurons:
            n.feedforward()
        
class MLP:
    def __init__(self, infile, dataset, print_step=100, verbose=True): # infile: MLP description file, dataset: Dataset object
        self.verbose = verbose
        self.inputs = dataset
        self.plot = list() # You can use this to make plots
        self.print_step = print_step # Print accuracy during training every print_step
        sample, self.gt = dataset.next_sample() # Initialize input and output of MLP
        self.layers = [sample] # First layer of MLP: inputs
        with open(infile) as f:
            for line in f:
                n_units = int(line.strip())
                self.layers.append(Layer(self.layers[-1], n_units)) # Create other layers
    
    def feedforward(self):
        for i in range(1, len(self.layers)):
            self.layers[i].feedforward()
            
    def __str__(self):
        sizes = list()
        for l in self.layers:
            sizes.append(len(l))
        return str(sizes)
    
    def backpropagate(self, learning_rate):
        self.compute_gradients()
        self.apply_gradients(learning_rate)
        
    def compute_gradients(self):
        # First compute derivatives for the last layer
        layer = self.layers[-1]
        for i in range(len(layer)):
            # Compute dL/du_i
            neuron = layer.neurons[i]
            o = neuron.z
            a = neuron.a
            t = self.gt[i]
            neuron.d_a = 2 * ( o - t ) * d_sigmoid(a) # Calculate the error function (o or sigmoid(u)?)
            # Compute dL/dw_ji
            neuron.d_weights = np.asscalar(neuron.d_a) * np.asarray([neuron.inputs[i] for i in range(len(neuron.inputs))]).reshape(-1,1)
    
        # Then compute derivatives for other layers
        for l in range(2, len(self.layers)):
            layer = self.layers[-l]
            next_layer = self.layers[-l+1]
            for i in range(len(layer)):
                # Compute dL/du_i
                neuron = layer.neurons[i]
                d_a = 0.
                a = neuron.a
                for j in range(len(next_layer)):
                    d_a += d_sigmoid(a) * next_layer.neurons[j].weights[i] * next_layer.neurons[j].d_a #OK
                neuron.d_a = d_a
                # Compute dL/dw_ji
            neuron.d_weights = np.asscalar(neuron.d_a) * np.asarray([neuron.inputs[i] for i in range(len(neuron.inputs))]).reshape(-1,1)
    
    def apply_gradients(self, learning_rate):
        # Change weights according to computed gradients
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            for j in range(1, len(layer)):
                neuron = layer.neurons[j]
                neuron.weights -= learning_rate * neuron.d_weights
                neuron.bias -= learning_rate * neuron.d_a
    
    def train_one_epoch(self, learning_rate):
        for i in range(len(self.inputs)):
            self.setnextinput() # Use next sample of dataset as next input
            self.feedforward() # Feed forward...
            self.backpropagate(learning_rate) # and backpropagate

    def train(self, n_epochs, learning_rate, decay=1.):
        # previous_weights = []
        # for l in self.layers[1:]:
        #     for n in l.neurons:
        #         previous_weights.append(n.weights)
        for i in range(n_epochs):
            self.train_one_epoch(learning_rate)
            if not (i+1)%(self.print_step):
                if self.verbose:
                    print("Epoch:", i+1, "out of", n_epochs)
                    self.print_accuracy()
                else:
                    self.compute_accuracy()
            learning_rate *= decay
        # new_weights = []
        # for l in self.layers[1:]:
        #     for n in l.neurons:
        #         new_weights.append(n.weights)
        
    def setnextinput(self):
        '''
        Set input of MLP to next input of dataset
        '''
        sample, gt = self.inputs.next_sample()
        self.gt = gt
        for i in range(len(self.layers[0])):
            self.layers[0][i] = sample[i]

    def save_MLP(self, filename):
        '''
        Not implemented yet
        '''
        pass

    def restore_MLP(self, filename):
        '''
        Not implemented yet
        '''
        pass

    def print_accuracy(self):
        print("Accuracy:", 100*self.compute_accuracy(), "%")

    def compute_accuracy(self):
        n_samples = len(self.inputs)
        n_accurate = 0.
        self.inputs.index = 0
        for i in range(n_samples):
            self.setnextinput()
            self.feedforward()
            if argmax(self.layers[-1]) == argmax(self.gt):
                n_accurate += 1.
        self.plot.append(n_accurate/n_samples)
        return n_accurate/n_samples

    def reset_plot(self):
        self.plot = list()

    def make_plot(self):
        plt.scatter([x*self.print_step for x in range(len(self.plot))], self.plot, color='r', marker='.')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch number')
        plt.show()
        

    def setdataset(self, dataset):
        self.inputs = dataset