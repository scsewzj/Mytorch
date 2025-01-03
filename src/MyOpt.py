from ComputeGraph import *
from MyTensor import *
from MyNN import *
import MyModel
import numpy as np

class myopt:

    def __init__(self,parameter, type='BGD', lr = 0.01):
        self.lr = lr
        self.type = type
        self.G = MyModel.mymodel.computegraph
        self.parameter = parameter

    def step(self):
        pass

    def zero_grad(self):
        self.G.zero_grad()

    def GSlim(self):
        self.G.slim()

class BGD(myopt):
    def __init__(self, parameter, lr = 0.1):
        super().__init__(parameter, lr = lr, type='BGD')

    def step(self, lr=None):
        if lr is not None:
            self.lr = lr
        for i in self.parameter:
            i._myparameter__update(self.lr)


class SGD(myopt):
    def __init__(self, parameter, lr = 0.1):
        super().__init__(parameter, lr = lr, type='SGD')

    def step(self):
        for i in self.G._parameternodes_index:
            self.G._nodelist[i].__update(self.lr)