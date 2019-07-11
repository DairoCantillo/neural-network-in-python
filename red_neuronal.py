# -*- coding: utf-8 -*-
"""Red Neuronal.ipynb
Deca
"""
import numpy as np
import matplotlib.pyplot as plt
import time
from train import *
from datasets.dataset import *
from activation_funtions import *
# CLASE DE LA  CAPA DE LA RED

class neural_layer():

    def __init__(self, n_connection, n_neuron, activation_funtion):

        self.activation_funtion = activation_funtion

        self.b = np.random.rand(1, n_neuron) * 2 - 1
        self.w = np.random.rand(n_connection, n_neuron) * 2 - 1


_x = np.linspace(-5, 5, 100)
#plt.plot(_x, sigm[0](_x))
#plt.plot(_x, relu(_x))

#layer0 = neural_layer(p, 4, sigm)


def create_red_neuronal(topology, activation_funtion):

    neuronal = []

    for l, layer in enumerate(topology[:-1]):

        neuronal.append(neural_layer(
            topology[l], topology[l+1], activation_funtion))
    return neuronal


topology = [p, 4, 8, 1]

neural_net = create_red_neuronal(topology, sigm)

l2_coste = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
            lambda Yp, Yr: (Yp - Yr))


train(neural_net, X, Y, l2_coste, 0.5)


