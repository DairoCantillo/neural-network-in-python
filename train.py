# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time

def train(neural_net, X, Y, l2_coste, lr=0.5, train=True):

    # forward pass

    out = [(None, X)]

    for l, layer in enumerate(neural_net):

        z = out[-1][1] @ neural_net[l].w + neural_net[l].b
        a = neural_net[l].activation_funtion[0](z)

        out.append((z, a))
    print(l2_coste[0](out[-1][1], Y))

    if train:
        # Backward pass
        deltas = []

        for l in reversed(range(0, len(neural_net))):

            z = out[l+1][0]
            a = out[l+1][1]
            print(a.shape)

            if l == len(neural_net)-1:
                # calcular delta de la ultima capa
                deltas.insert(0, l2_coste[1](
                    a, Y) * neural_net[l].activation_funtion[1](a))
                # print(deltas.shape)
            else:
                # calcular delta respecto a capa previa
                deltas.insert(0, deltas[0] @ _w.T *
                              neural_net[l].activation_funtion[1](a))

            _w = neural_net[l].w

            # Gradient descent
            neural_net[l].b = neural_net[l].b - \
                np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].w = neural_net[l].w - out[l][1].T @ deltas[0] * lr
    return out[-1][1]