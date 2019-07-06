# -*- coding: utf-8 -*-
"""Red Neuronal.ipynb
Deca
"""

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

#CREAR EL DATASET

n = 500
p = 2

X, Y = make_circles(n_samples = n, factor = 0.5, noise=.05)

Y = Y[:, np.newaxis]

plt.scatter(X[Y[:, 0]==0, 0], X[Y[:, 0]==0, 1], c="skyblue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c ="salmon")
plt.axis("equal")
plt.show()

#CLASE DE LA  CAPA DE LA RED

class neural_layer():
  
  def __init__(self, n_connection,n_neuron, activation_funtion):
    
    self.activation_funtion = activation_funtion
    
    self.b = np.random.rand(1, n_neuron) * 2 - 1
    self.w = np.random.rand(n_connection, n_neuron) * 2 - 1

#FUNCION DE ACTIVACION

sigm = (lambda x:1/(1+np.e**(-x)), 
        lambda x:x*(1-x))

relu = lambda x: np.maximum(0,x)

_x = np.linspace(-5,5,100)
plt.plot(_x, sigm[0](_x))
plt.plot(_x, relu(_x))

#layer0 = neural_layer(p, 4, sigm)




def create_red_neuronal(topology, activation_funtion):
  
  neuronal = []
  
  for l, layer in enumerate(topology[:-1]):
    
    neuronal.append(neural_layer(topology[l],topology[l+1], activation_funtion))
  return neuronal

topology = [p, 4, 8, 1]

neural_net = create_red_neuronal(topology, sigm)

l2_coste = (lambda Yp, Yr: np.mean((Yp - Yr) ** 2),
            lambda Yp, Yr:(Yp - Yr))

def train(neural_net, X, Y, l2_coste, lr=0.5, train=True):
  
  #forward pass
 
  out = [(None, X)]
  
  for l, layer in enumerate(neural_net):
    
    z = out[-1][1] @ neural_net[l].w + neural_net[l].b
    a = neural_net[l].activation_funtion[0](z)
                                         
    out.append((z, a))
  print(l2_coste[0](out[-1][1], Y))
  
  if train:
    #Backward pass
    deltas = []
    
    for l in reversed(range(0, len(neural_net))):
      
      z = out[l+1][0]
      a = out[l+1][1]
      print(a.shape)
      
      if l==len(neural_net)-1 :
        #calcular delta de la ultima capa
        deltas.insert(0, l2_coste[1](a, Y)* neural_net[l].activation_funtion[1](a))
        #print(deltas.shape)
      else:
        #calcular delta respecto a capa previa
        deltas.insert(0, deltas[0] @ _w.T * neural_net[l].activation_funtion[1](a))
        
      _w = neural_net[l].w
      
      #Gradient descent
      neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) *lr
      neural_net[l].w = neural_net[l].w - out[l][1].T @ deltas[0] *lr
  return out[-1][1]
      
    
train(neural_net, X, Y, l2_coste, 0.5)

# VISUALIZACIÓN Y TEST

import time
from IPython.display import clear_output

neural_n = create_red_nueronal(topology, sigm)

loss = []

for i in range(1000):
    
  # Entrenemos a la red!
  pY = train(neural_n, X, Y, l2_coste, lr=0.05)
  
  if i % 25 == 0:
    
    print(pY)
  
    loss.append(l2_coste[0](pY, Y))
  
    res = 50

    _x0 = np.linspace(-1.5, 1.5, res)
    _x1 = np.linspace(-1.5, 1.5, res)

    _Y = np.zeros((res, res))

    for i0, x0 in enumerate(_x0):
      for i1, x1 in enumerate(_x1):
        _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_coste, train=False)[0][0]    

    plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
    plt.axis("equal")

    plt.scatter(X[Y[:,0] == 0, 0], X[Y[:,0] == 0, 1], c="skyblue")
    plt.scatter(X[Y[:,0] == 1, 0], X[Y[:,0] == 1, 1], c="salmon")

    clear_output(wait=True)
    plt.show()
    plt.plot(range(len(loss)), loss)
    plt.show()
    time.sleep(0.5)