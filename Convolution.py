import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import math
from scipy.stats import norm
import sys
import pickle
import gzip
import h5py
from tensorflow.keras.datasets import mnist
import tensorflow
import timeit

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_mnist_dataset():
    f = gzip.open('mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')
    f.close()

    return data

#Activation Function
class tanh:
    @staticmethod
    def activation(x):
        y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return y

    @staticmethod
    def prime(x):
        y = 1 - (tanh.activation(x)**2)
        return y

class sigmoid:
    @staticmethod
    def activation(x):
        y = 1 / (1 + np.exp(-x))
        return y

    @staticmethod
    def prime(x):
        y = sigmoid.activation(x) * (1 - sigmoid.activation(x))
        return y

class relu:
    @staticmethod
    def activation(x):
        y = np.maximum(0, x)
        return y

    @staticmethod
    def prime(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

class softmax:
    @staticmethod
    def activation(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)

class BinaryCrossEntropy:

    @staticmethod
    def call(m, y, output):
        return (-1) * (1 / m) * (np.sum((y * np.log(output+1e-8)) + ((1 - y) * (np.log(1 - output+1e-8)))))

    @staticmethod
    def prime(m, y1, y2):
        return (-1 / m) * ((y1 / y2) + (y1 - 1) * (1 / (1 - y2)))

class CategoricalCrossEntropy:

    @staticmethod
    def call(m, y, output):
        return (-1) * (1 / m) * np.sum((y * np.log(output)))

    @staticmethod
    def prime(m, y1, y2):
        return (-1 / m) * (y1 / y2)

class Initialization:

    @staticmethod
    def Zeros(X_train, layers):
        input = X_train
        layers = layers

        w = {}
        sdw = {}
        vdw = {}

        w[1] = np.zeros((input.shape[0], model.get_neurons(1)))
        sdw[1] = np.zeros((input.shape[0], model.get_neurons(1)))
        vdw[1] = np.zeros((input.shape[0], model.get_neurons(1)))

        for i in range(layers - 1):
            w[i + 2] = np.zeros((model.get_neurons(i + 1), model.get_neurons(i + 2)))
            sdw[i + 2] = np.zeros((input.shape[0], model.get_neurons(1)))
            vdw[i + 2] = np.zeros((input.shape[0], model.get_neurons(1)))

        b = {}
        sdb = {}
        vdb = {}

        for i in range(layers):
            b[i + 1] = np.zeros((model.get_neurons(i + 1), 1))
            sdb[i + 1] = np.zeros((model.get_neurons(i + 1), 1))
            vdb[i + 1] = np.zeros((model.get_neurons(i + 1), 1))

        return w, b, sdw, sdb, vdw, vdb

    @staticmethod
    def Xavier(X_train, layers):
        input = X_train
        layers = layers

        w = {}
        sdw = {}
        vdw = {}

        w[1] = np.random.randn(input.shape[0], model.get_neurons(1)) * np.sqrt(1 / input.shape[0] )
        sdw[1] = np.zeros((input.shape[0], model.get_neurons(1)))
        vdw[1] = np.zeros((input.shape[0], model.get_neurons(1)))

        for i in range(layers - 1):
            w[i + 2] = np.random.randn(model.get_neurons(i + 1), model.get_neurons(i + 2)) * np.sqrt(
                1 / model.get_neurons(i + 1))

            sdw[i + 2] = np.zeros((model.get_neurons(i + 1), model.get_neurons(i + 2)))
            vdw[i + 2] = np.zeros((model.get_neurons(i + 1), model.get_neurons(i + 2)))

        b = {}
        sdb = {}
        vdb = {}

        for i in range(layers):
            b[i + 1] = np.zeros((model.get_neurons(i + 1), 1))
            sdb[i + 1] = np.zeros((model.get_neurons(i + 1), 1))
            vdb[i + 1] = np.zeros((model.get_neurons(i + 1), 1))

        return w, b, sdw, sdb, vdw, vdb

    @staticmethod
    def He(X_train, layers):
        input = X_train
        layers = layers

        w = {}
        sdw = {}
        vdw = {}

        w[1] = np.random.randn(input.shape[0], model.get_neurons(1)) * np.sqrt(2 / input.shape[0])
        sdw[1] = np.zeros((input.shape[0], model.get_neurons(1)))
        vdw[1] = np.zeros((input.shape[0], model.get_neurons(1)))

        for i in range(layers - 1):
            w[i + 2] = np.random.randn(model.get_neurons(i + 1), model.get_neurons(i + 2)) * np.sqrt(
                2 / model.get_neurons(i + 1))

            sdw[i + 2] = np.zeros((model.get_neurons(i + 1), model.get_neurons(i + 2)))
            vdw[i + 2] = np.zeros((model.get_neurons(i + 1), model.get_neurons(i + 2)))

        b = {}
        sdb = {}
        vdb = {}

        for i in range(layers):
            b[i + 1] = np.zeros((model.get_neurons(i + 1), 1))
            sdb[i + 1] = np.zeros((model.get_neurons(i + 1), 1))
            vdb[i + 1] = np.zeros((model.get_neurons(i + 1), 1))

        return w, b, sdw, sdb, vdw, vdb

    @staticmethod
    def Kumar(X_train, layers):
        input = X_train
        layers = layers

        w = {}
        sdw = {}
        vdw = {}

        w[1] = np.random.randn(input.shape[0], model.get_neurons(1)) * np.sqrt(12.96 / input.shape[0])
        sdw[1] = np.zeros((input.shape[0], model.get_neurons(1)))
        vdw[1] = np.zeros((input.shape[0], model.get_neurons(1)))

        for i in range(layers - 1):
            w[i + 2] = np.random.randn(model.get_neurons(i + 1), model.get_neurons(i + 2)) * np.sqrt(
                12.96 / model.get_neurons(i + 1))

            sdw[i + 2] = np.zeros((model.get_neurons(i + 1), model.get_neurons(i + 2)))
            vdw[i + 2] = np.zeros((model.get_neurons(i + 1), model.get_neurons(i + 2)))

        b = {}
        sdb = {}
        vdb = {}

        for i in range(layers):
            b[i + 1] = np.zeros((model.get_neurons(i + 1), 1))
            sdb[i + 1] = np.zeros((model.get_neurons(i + 1), 1))
            vdb[i + 1] = np.zeros((model.get_neurons(i + 1), 1))

        return w, b, sdw, sdb, vdw, vdb

class Learning_Rate_Schedules():

    @staticmethod
    def exp_decay(learning_rate, decay, currentepoch):

        lrate = learning_rate * math.exp(-decay * currentepoch)

        if lrate <= 0.001:
            lrate = 0.001

        return lrate

    @staticmethod
    def time_based_decay(learning_rate, decay, currentepoch):

        learning_rate *= (1.0 / (1.0 + decay * currentepoch))

        if learning_rate <= 0.001:
            learning_rate = 0.001

        return learning_rate

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

class CNN:
    def __init__(self):
        self.Loss_list = []
        self.epochs_list = []
        self.accuracy_values = []
        self.neurons = []
        self.activations = {}
        self.layers = 0
        self.activationMean = []
        self.gradientMean = []

    @staticmethod
    def flatten(x):
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).T

    def GaussianBlur_1(self):
        a = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]])
        k = a / np.sum(a)
        return k

    def edge_detection(self):
        a = np.array([[1, 0, -1],
                      [0, 0, 0],
                      [-1, 0, 1]])
        return a

    def sharpen(self):
        a = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
        return a

    def GaussianBlur_2(self):
        a = (1 / 256) * np.array([[1, 4, 6, 4, 1],
                                  [4, 16, 24, 16, 4],
                                  [6, 24, 36, 24, 6],
                                  [4, 16, 24, 16, 4],
                                  [1, 4, 6, 4, 1]])
        return a

    def sobel_edge_1(self):
        a = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]])
        return a

    def sobel_edge_2(self):
        a = np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]]).T
        return a

    def laplacian(self):
        a = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]])
        return a

    def Laplacian_of_Gaussian(self):
        a = np.array([[0, 0, -1, 0, 0],
                      [0, -1, -2, -1, 0],
                      [-1, -2, 16, -2, -1],
                      [0, -1, -2, -1, 0],
                      [0, 0, -1, 0, 0]])
        return a

    def add_layer(self, neurons, activation):
        self.neurons.append(neurons)
        self.layers += 1

        self.activations[self.layers] = activation

        for i in range(self.layers):
            array = []
            self.activationMean.append(array)

        for i in range(self.layers):
            array = []
            self.gradientMean.append(array)

    def num_layers(self):
        return print('Total number of layers: ' + str(self.layers))

    def get_layers_list(self):
        return self.layers

    def get_neurons_list(self):
        return self.neurons

    def get_neurons(self, layer):
        return self.neurons[layer - 1]

    def get_layer_info(self, num):
        a = self.neurons[num - 1]
        b = self.activations[num - 1]
        return a, b

    def complile(self, loss, initialization, optimizer):
        self.cost = loss
        self.initialization = initialization
        self.optimizer = optimizer

    def acc(self, y_true, y_predicted, cost):
        if cost == 'BinaryCrossEntropy':
            accuracy = np.mean(np.equal(y_true, np.round(y_predicted))) * 100
        else:
            accuracy = np.mean(np.equal(np.argmax(y_true, axis=-1), np.argmax(y_predicted, axis=-1))) * 100
        return accuracy

    def activation_mean(self):

        x = np.arange(1, model.epochs + 1)

        for i in range(0, self.layers):
            layer = self.activationMean[i]
            plt.plot(x, layer, label="Layer {}".format(i + 1))

        # plt.ylim(0.1, 0.9)
        plt.xlabel('epochs')
        plt.ylabel('Activation mean')
        plt.legend(loc='upper right')

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('activation_mean.png')

    def activation_distribution(self):

        global x1, x2

        if self.activations[1] == 'sigmoid':
            x1 = 0
            x2 = 1

        if self.activations[1] == 'tanh':
            x1 = -1
            x2 = 1

        for i in range(self.layers):
            mu, sigma = norm.fit(self.a[i + 1])
            dist = norm(mu, sigma)
            values = np.linspace(x1, x2, 500)
            probabilities = [dist.pdf(value) for value in values]
            plt.plot(values, probabilities, label="Layer {}".format(i + 1))

        plt.xlabel('Activation Value')
        plt.legend(loc='upper right')

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('activation_distribution.png')

    def z_distribution(self):

        for i in range(self.layers):
            mu, sigma = norm.fit(self.z[i + 1])
            dist = norm(mu, sigma)
            values = np.linspace(-2, 2, 500)
            probabilities = [dist.pdf(value) for value in values]
            plt.plot(values, probabilities, label="Layer {}".format(i + 1))

        plt.xlabel('Z Distribution')
        plt.legend(loc='upper right')

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('Z_Distribution.png')

    def backpropogation_gradients_distribution(self):

        weight_gradients = self.update_params

        for i in range(1, self.layers + 1):
            an_array = weight_gradients[i][0]
            norm_ = np.linalg.norm(an_array)
            update_params = an_array / norm_

            mu, sigma = norm.fit(update_params)
            dist = norm(mu, sigma)
            values = np.linspace(-0.2, 0.2, 100)
            probabilities = [dist.pdf(value) for value in values]
            plt.plot(values, probabilities, label="Layer {}".format(i))

        plt.xlabel('Backpropogation gradients')
        plt.legend(loc='upper right')

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('backpropogation_distribution.png')

    def gradient_mean(self):

        x = np.arange(1, model.epochs + 1)

        for i in range(0, self.layers):
            layer = self.gradientMean[i]
            plt.plot(x, layer, label="Layer {}".format(i + 1))

        plt.ylim(-0.00001, 0.0001)
        plt.xlabel('epochs')
        plt.ylabel('gradient mean')
        plt.legend(loc='upper right')

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('backpropogation_mean.png')

    def GDScheduler(self, lr, momemtum=None, decay=None, method=None):
        self.learning_rate = lr
        self.decay = decay
        self.momemtum = momemtum

        if method == None:
            self.decaymethod = 'None'

        if method == 'exponential':
            self.decaymethod = 'exp_decay'

        if method == 'timebased':
            self.decaymethod = 'time_based_decay'

    def GD(self, index, dw, db):

        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * db

    def RMSprop(self, index, gamma, dw, db):

        self.sdw[index] = gamma * self.sdw[index] + (1 - gamma) * dw**2
        self.sdb[index] = gamma * self.sdb[index] + (1 - gamma) * db**2

        self.w[index] -= (self.learning_rate / (np.sqrt(self.sdw[index]+1e-08))) * dw
        self.b[index] -= (self.learning_rate / (np.sqrt(self.sdb[index]+1e-08))) * db

    def Adam(self, index, gamma1, gamma2, dw, db):

        vdw_corr = {}
        vdb_corr = {}

        sdw_corr = {}
        sdb_corr = {}

        self.vdw[index] = gamma1 * self.vdw[index] + (1 - gamma1) * dw
        self.vdb[index] = gamma1 * self.vdb[index] + (1 - gamma1) * db

        self.sdw[index] = gamma2 * self.sdw[index] + (1 - gamma2) * dw**2
        self.sdb[index] = gamma2 * self.sdb[index] + (1 - gamma2) * db**2

        vdw_corr[index] = self.vdw[index] / (1 - np.power(gamma1, self.currentepoch+1))
        vdb_corr[index] = self.vdb[index] / (1 - np.power(gamma1, self.currentepoch+1))

        sdw_corr[index] = self.sdw[index] / (1 - np.power(gamma2, self.currentepoch+1))
        sdb_corr[index] = self.sdb[index] / (1 - np.power(gamma2, self.currentepoch+1))

        self.w[index] -= (self.learning_rate / (np.sqrt(sdw_corr[index]+1e-08))) * vdw_corr[index]
        self.b[index] -= (self.learning_rate / (np.sqrt(sdb_corr[index]+1e-08))) * vdb_corr[index]

    def Adamax(self, index, gamma1, gamma2, dw, db):

        vdw_corr = {}
        vdb_corr = {}

        self.vdw[index] = gamma1 * self.vdw[index] + (1 - gamma1) * dw
        self.vdb[index] = gamma1 * self.vdb[index] + (1 - gamma1) * db

        self.sdw[index] = np.maximum(gamma2 * self.sdw[index], np.abs(dw))
        self.sdb[index] = np.maximum(gamma2 * self.sdb[index], np.abs(db))

        vdw_corr[index] = self.vdw[index] / (1 - np.power(gamma1, self.currentepoch+1))
        vdb_corr[index] = self.vdb[index] / (1 - np.power(gamma1, self.currentepoch+1))

        self.w[index] -= (self.learning_rate / (np.sqrt(self.sdw[index]+1e-08))) * vdw_corr[index]
        self.b[index] -= (self.learning_rate / (np.sqrt(self.sdb[index]+1e-08))) * vdb_corr[index]

    def feedforward(self, j):

        global cost
        self.z = {}
        self.a = {}

        self.a = {0: self.input}

        # Initialize parameters
        if j == 0:
            if self.initialization == 'Xavier':
                self.w, self.b, self.sdw, self.sdb, self.vdw, self.vdb = Initialization.Xavier(self.input, self.layers)
            elif self.initialization == 'He':
                self.w, self.b, self.sdw, self.sdb, self.vdw, self.vdb = Initialization.He(self.input, self.layers)
            elif self.initialization == 'Kumar':
                self.w, self.b, self.sdw, self.sdb, self.vdw, self.vdb = Initialization.Kumar(self.input, self.layers)
            else:
                self.w, self.b, self.sdw, self.sdb, self.vdw, self.vdb = Initialization.Zeros(self.input, self.layers)

        # CostFunction
        if self.cost == 'BinaryCrossEntropy':
            cost = 'BinaryCrossEntropy'

        if self.cost == 'CategoricalCrossEntropy':
            cost = 'CategoricalCrossEntropy'

        for i in range(0, self.layers):
            self.z[i + 1] = np.dot(self.w[i + 1].T, self.a[i]) + self.b[i + 1]
            self.a[i + 1] = eval(self.activations[i + 1]).activation(self.z[i + 1])
            self.activationMean[i].append(self.a[i + 1].mean())

        self.output = self.a[self.layers]

        self.loss = eval(cost).call(self.m, self.y, self.output)

        return self.z, self.a, self.output, self.loss

    def backpropogation(self):

        delta = self.output - self.y
        dw = (1 / self.m) * np.dot(delta, self.a[self.layers - 1].T).T
        db = (1 / self.m) * np.sum(delta)

        update_params = {
            self.layers: (dw, db)
        }

        self.gradientMean[self.layers - 1].append(abs(dw.mean()))

        for i in reversed(range(1, self.layers)):
            delta = np.dot(self.w[i + 1].T.T, delta) * eval(self.activations[i]).prime(self.z[i])
            dw = (1 / self.m) * np.dot(delta, self.a[i - 1].T).T
            db = (1 / self.m) * np.sum(delta)

            # Storing dw and db
            update_params[i] = (dw, db)

            # Storing gradient mean values
            self.gradientMean[i - 1].append(abs(dw.mean()))

        # Optimizer
        if self.optimizer == 'GD':
            for i, j in update_params.items():
                self.GD(i, j[0], j[1])

        if self.optimizer == 'RMSprop':
            for i, j in update_params.items():
                self.RMSprop(i, 0.9, j[0], j[1])

        if self.optimizer == 'Adam':
            for i, j in update_params.items():
                self.Adam(i, 0.9, 0.999, j[0], j[1])

        if self.optimizer == 'Adamax':
            for i, j in update_params.items():
                self.Adamax(i, 0.9, 0.999, j[0], j[1])

        return update_params

    def propogation(self, i):
        self.z, self.a, self.output, self.loss = self.feedforward(i)
        #self.update_params = self.backpropogation()
        return self.z, self.a, self.output, self.loss, #self.update_params

    def fit(self, X_train, y_train, epochs):

        self.input = X_train
        self.y = y_train
        self.m = X_train.shape[1]
        self.epochs = epochs

        print("Training........")
        for i in range(self.epochs):

            self.currentepoch = i

            if self.decaymethod == 'None':
                pass

            if self.decaymethod == 'exp_decay':
                self.learning_rate = Learning_Rate_Schedules.exp_decay(self.learning_rate, self.decay,
                                                                       self.currentepoch)

            if self.decaymethod == 'time_based_decay':
                self.learning_rate = Learning_Rate_Schedules.time_based_decay(self.learning_rate, self.decay,
                                                                              self.currentepoch)

            start = timeit.default_timer()

            for j in range(self.input.shape[1]):

                self.z, self.a, self.output, self.loss = self.propogation(i)

                if self.cost == 'CategoricalCrossEntropy':
                    probablity = self.output.T
                    y_predicted = np.zeros_like(probablity)
                    y_predicted[np.arange(len(probablity)), probablity.argmax(1)] = 1
                    y_trues = self.y.T

                    self.accuracy = self.acc(y_trues, y_predicted, self.cost)
                else:
                    self.accuracy = self.acc(self.y, self.output, self.cost)

            end = timeit.default_timer()

            print("epochs:" + str(i) + " | "
                  "runtime: {} s".format(float(round(end-start, 3))) + " | "
                  "Loss:" + str(self.loss) + " | "
                  "Accuracy: {} %".format(float(round(self.accuracy, 3))))

            if i % 1 == 0:
                self.accuracy_values.append(self.accuracy)
                self.Loss_list.append(self.loss)
                self.epochs_list.append(i)

            if i == 0:
                self.a1 = self.a

        # accuracy Plot
        accuracy_list = np.array(self.accuracy_values)
        accuracy_list = accuracy_list.reshape(-1, 1)

        # Loss Plot
        Loss_array = np.array(self.Loss_list)
        y_loss = Loss_array.reshape(-1, 1)
        x_epochs = np.array(self.epochs_list).reshape(-1, 1)

        accuracy_data = pd.DataFrame()
        accuracy_data['0'] = x_epochs.reshape(1, -1)[0]
        accuracy_data['1'] = accuracy_list.reshape(1, -1)[0]
        accuracy_data.to_csv('accuracy.txt', index=False, header=False, sep=" ")

        loss_data = pd.DataFrame()
        loss_data['0'] = x_epochs.reshape(1, -1)[0]
        loss_data['1'] = y_loss.reshape(1, -1)[0]
        loss_data.to_csv('cost.txt', index=False, header=False, sep=" ")

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x_epochs, accuracy_list)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('epochs_vs_accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(x_epochs, y_loss)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('epochs_vs_loss')

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('Results.png')

        print("Training accuracy: {} %".format(self.accuracy))

    def predict(self, x, threshold=None):

       self.input = x
       self.z = {}
       self.a = {}

       self.a = {0: self.input}

       for i in range(0, self.layers):
           self.z[i + 1] = np.dot(self.w[i + 1].T, self.a[i]) + self.b[i + 1]
           self.a[i + 1] = eval(self.activations[i + 1]).activation(self.z[i + 1])

       self.output = self.a[self.layers]

       if self.cost == 'BinaryCrossEntropy':
            probablity = self.output

            probablity[probablity <= threshold] = 0
            probablity[probablity > threshold] = 1

            y_predicted = probablity.astype(int)

       else:

           probablity = self.output.T

           y_predicted = np.zeros_like(probablity)
           y_predicted[np.arange(len(probablity)), probablity.argmax(1)] = 1
           y_predicted = y_predicted.T

       return y_predicted

    def confusion_matrix(self, data_array, labels):

        dim = len(data_array[0])
        cm = np.zeros((dim, dim), int)

        for i in range(len(data_array)):
            truth = np.argmax(data_array[i])
            predicted = np.argmax(labels[i])
            cm[truth, predicted] += 1

        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='d')
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('cm.png')
        return cm

    def evaluate(self, y_test, y_predicted):
        dim = len(y_test[0])
        cm = np.zeros((dim, dim), int)

        for i in range(len(y_test)):
            truth = np.argmax(y_test[i])
            predicted = np.argmax(y_predicted[i])
            cm[truth, predicted] += 1

        accuracy = np.sum(cm.diagonal()) / np.sum(cm)
        print("Testing accuracy: {} %".format(accuracy * 100))

    def precision(self, y_test, y_predicted):

        precision = []

        dim = len(y_test[0])
        cm = np.zeros((dim, dim), int)

        for i in range(len(y_test)):
            truth = np.argmax(y_test[i])
            predicted = np.argmax(y_predicted[i])
            cm[truth, predicted] += 1

        for i in range(len(y_test)):
            col = cm[:, i]
            precision.append(cm[i, i] / col.sum())

        return precision

    def recall(self, y_test, y_predicted):

        recall = []

        dim = len(y_test[0])
        cm = np.zeros((dim, dim), int)

        for i in range(len(y_test)):
            truth = np.argmax(y_test[i])
            predicted = np.argmax(y_predicted[i])
            cm[truth, predicted] += 1

        for i in range(len(y_test)):
            row = cm[i, :]
            recall.append(cm[i, i] / row.sum())

        return recall

    def X_flatten(self, X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=0)

        windows = []
        for i in range(out_h):
            for j in range(out_w):
                window = X_padded[:, i * stride:i * stride + window_h, j * stride:j * stride + window_w, :]
                windows.append(window)
        stacked = np.stack(windows)  # shape : [out_h, out_w, n, filter_h, filter_w, c]

        return np.reshape(stacked, (-1, window_c * window_w * window_h))

    def convolution(self, X, n_filters, kernel_size, padding, stride, activation=None):

        global conv_activation_layer
        k_h = kernel_size[0]
        k_w = kernel_size[1]

        if padding == 'VALID':
            pad = 0
        else:
            pad = 1

        filters = []
        for i in range(n_filters):
            kernel = np.random.randn(k_h, k_w, X.shape[3])
            filters.append(kernel)
        kernel = np.reshape(filters, (k_h, k_w, X.shape[3], n_filters))

        n, h, w, c = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        filter_h, filter_w, filter_c, filter_n = kernel.shape[0], kernel.shape[1], kernel.shape[2], kernel.shape[3]

        out_h = (h + 2 * pad - filter_h) // stride + 1
        out_w = (w + 2 * pad - filter_w) // stride + 1

        X_flat = model.X_flatten(X, filter_h, filter_w, filter_c, out_h, out_w, stride, pad)
        W_flat = np.reshape(kernel, (filter_h * filter_w * filter_c, filter_n))

        z = np.matmul(X_flat, W_flat)
        z = np.transpose(np.reshape(z, (out_h, out_w, n, filter_n)), (2, 0, 1, 3))

        if activation == 'relu':
            conv_activation_layer = relu.activation(z)

        return conv_activation_layer

    def max_pool(self, X, pool_size, padding, stride):

        pool_h = pool_size[0]
        pool_w = pool_size[1]

        if padding == 'VALID':
            pad = 0
        else:
            pad = 1

        n, h, w, c = X.shape[0], X.shape[1], X.shape[2], X.shape[3]

        out_h = (h + 2 * pad - pool_h) // stride + 1
        out_w = (w + 2 * pad - pool_w) // stride + 1

        X_flat = model.X_flatten(X, pool_h, pool_w, c, out_h, out_w, stride, pad)

        pool = np.max(np.reshape(X_flat, (out_h, out_w, n, pool_h * pool_w, c)), axis=3)

        return np.transpose(pool, (2, 0, 1, 3))

if __name__ == '__main__':

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    plt.imshow(X_train[0], cmap=plt.get_cmap('gray_r'))
    plt.show()

    X_train = X_train / 255
    X_test = X_test / 255

    y_train = tensorflow.keras.utils.to_categorical(y_train, 10).T
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10).T

    X_train = X_train.reshape((60000, 28, 28, 1))

    training = X_train[:5000]
    y_train = y_train[:, :5000]

    print(training.shape)

    model = CNN()

    Layer_1 = model.convolution(training, n_filters=6, kernel_size=(5, 5), padding='VALID', stride=1, activation='relu')

    print(Layer_1.shape)

    Layer_2 = model.max_pool(Layer_1, pool_size=(2, 2), padding='VALID', stride=2)

    print(Layer_2.shape)

    Layer_3 = model.convolution(Layer_2, n_filters=16, kernel_size=(5, 5), padding='VALID', stride=1, activation='relu')

    print(Layer_3.shape)

    Layer_4 = model.max_pool(Layer_3, pool_size=(2, 2), padding='VALID', stride=2)

    print(Layer_4.shape)

    Layer_5 = model.flatten(Layer_4)

    print(Layer_5.shape)

    model.add_layer(120, activation='relu')
    model.add_layer(10, activation='softmax')

    model.complile(loss='CategoricalCrossEntropy', initialization='He', optimizer='Adam')
    model.GDScheduler(lr=0.001)

    ##X_train = (x*y, m), y_train = (10, m)
    start = timeit.default_timer()
    model.fit(Layer_5, y_train, epochs=1)
    stop = timeit.default_timer()