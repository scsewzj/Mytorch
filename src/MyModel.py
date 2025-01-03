from MyNN import *
from MyOpt import *
from MyPara import *
from MyTensor import *
from ComputeGraph import *
from util import *


class mymodel:

    computegraph = computegraph()

    def __init__(self, modelshape : list = None, tensortype = 'tensor32', linear=True):
        self.layers=[]
        if modelshape is not None and linear:
            for i in modelshape:
                self.layers.append(MyNN.my_linear_layer(i, tensortype = tensortype))
            for i in range(len(self.layers)-1):
                self.layers[i].attach_parameterw(int(self.layers[i].size),int(self.layers[i+1].size))
                self.layers[i].attach_parameterb(int(self.layers[i+1].size))

    def __call__(self, X: mytensor):
        for i in range(len(self.layers)-1):
            X = self.layers[i](X)
        return X
        

    def save(self, path):
        pass

    def load(self, path):
        pass

    def train(self, datasource, batch_size = 32, L = 'L2', iterations = 1000, lr = 0.01, optmizer = 'default'):
        if optmizer == 'default':
            opt = myopt(lr = lr, L = L, type='grad_des')
            for i in range(iterations):
                X, label = dataloader(datasource, batch_size=batch_size)
                X = self(X)
                opt.Loss(X, label)
                opt.zero_grad()
                opt.backward()
                opt.step()
        else:
            return 0

    def test():
        pass
            