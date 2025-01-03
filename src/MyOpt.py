from ComputeGraph import *
from MyTensor import *
from MyNN import *
import MyModel
import numpy as np

class myopt:

    def __init__(self, lr = 0.01, L = 'L2', type='grad_des'):
        self.L = L
        self.lr = lr
        self.type = type
        self.loss = None
        self.G = MyModel.mymodel.computegraph

    def Loss(self, pred, label):
        oh_labels = []
        for i in label:
            ohi=np.zeros(len(pred.npar_data[0]))
            ohi[i]=1
            oh_labels.append(ohi)
        ohi = np.array(ohi)
        if self.L == 'L2':
            self.loss = np.linalg.norm(oh_labels - pred.npar_data)
        elif self.L == 'L1':
            self.loss = np.sum(np.abs(label - pred))
        else:
            raise ValueError('loss specified is not supported')

    def backward(self):
        self.G.__deletediscrete()
        shapeout = self.G._edgelist[-1]['to'].npar_data.shape
        L = np.array([self.loss for i in range(shapeout[0]*shapeout[1])]).reshape(shapeout)
        Lw = L
        Lb = L
        for i in range(len(self.G._edgelist)):
            Lw, Lb = self.G._edgelist[len(self.G._edgelist)-i-1]['forward'].backward(Lw, Lb)
            ### adjusted loss

    def zero_grad(self):
        self.G.__deletediscrete__()
        for i in self.G._nodelist:
            i.reinit_grad

    def step(self):
        for i in range(len(self.G._edgelist)):
            self.G._edgelist[len(self.G._edgelist)-i-1]['forward'].update_parameter(self.lr)