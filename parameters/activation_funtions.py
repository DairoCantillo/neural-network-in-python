# -*- coding: utf-8 -*-
import numpy as np


# FUNCIONes DE ACTIVACION
sigm = (lambda x: 1/(1+np.e**(-x)),
        lambda x: x*(1-x))

relu = (lambda x:np.maximum(0, x),
        lambda x:np.maximum(0, x), )

tanh = (lambda x: (2/(1+np.e**(-x)))-1,
        lambda x: x*(1-x))


#def relu(x): return np.maximum(0, x)

