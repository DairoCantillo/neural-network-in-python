# -*- coding: utf-8 -*-
from backpropagation import *
from IPython.display import clear_output
from parameters.activation_funtions import *
from parameters.topology import *

if __name__ == "__main__":
    # VISUALIZACIÓN Y TEST
    neural_n = create_red_neuronal(topology, sigm)

    loss = []

    for i in range(max_iter):

        # Entrenemos a la red!
        pY = train(neural_n, X, Y, l2_coste, lr=learning_rate)

        if i % 25 == 0:

            print(pY)

            loss.append(l2_coste[0](pY, Y))

            res = 50

            _x0 = np.linspace(-1.5, 1.5, res)
            _x1 = np.linspace(-1.5, 1.5, res)

            _Y = np.zeros((res, res))

            for i0, x0 in enumerate(_x0):
                for i1, x1 in enumerate(_x1):
                    _Y[i0, i1] = train(neural_n, np.array(
                        [[x0, x1]]), Y, l2_coste, train=False)[0][0]

    plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
    plt.axis("equal")

    plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
    plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")

    clear_output(wait=True)
    plt.show()
    plt.plot(range(len(loss)), loss)
    plt.show()

