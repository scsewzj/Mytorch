import MyTensor
import numpy as np

mytensor = MyTensor.mytensor

class loss:
    def __init__(self, comment=None):
        self.comment = comment
    def _loss(self, pred, true):
        pass
    def __call__(self, pred, true):
        self._tensorloss = self._loss(pred, true)
        self._losscursor_CG()
        return self._avg()

    def _avg(self):
        self.scalar = np.average(self._tensorloss.npar_data)
        return self.scalar

    def __repr__(self):
        if self.comment is None:
            return str(type(self))+': '+str(self.scalar)
        else:
            return str(type(self))+str(self.comment)+': '+str(self.scalar)

    def __str__(self):
        if self.comment is None:
            return str(type(self))+': '+str(self.scalar)
        else:
            return str(type(self))+str(self.comment)+': '+str(self.scalar)

    def _losscursor_CG(self):
        G = mytensor.computegraph
        for i in G._nodelist:
            if i==self._tensorloss and G._nodelist.index(i) not in G._lossnodes_index:
                G._lossnodes_index.append(G._nodelist.index(i))



class L1(loss):
    def __init__(self, comment=None):
        super().__init__(comment)

    def _loss(self, pred: mytensor, true: mytensor):
        if len(pred.shape)!=2 or len(true.shape)!=2:
            raise ValueError('Only Support Dim: Batchsize * out_features, transfer to one-hot code or expand input data to 2-D (including batchsize)')
            return False
        losstensor = (MyTensor.add(pred,-1*true) ** 2) ** 0.5
        sumup_mask = MyTensor.ones((pred.shape[1],1))
        return MyTensor.dot(losstensor,sumup_mask)

class L2(loss):
    def __init__(self, comment=None):
        super().__init__(comment)

    def _loss(self, pred: mytensor, true: mytensor):
        if len(pred.shape)!=2 or len(true.shape)!=2:
            raise ValueError('Only Support Dim: Batchsize * out_features, transfer to one-hot code or expand input data to 2-D (including batchsize)')
            return False
        losstensor = (MyTensor.add(pred,true*(-1))) ** 2
        sumup_mask = MyTensor.ones((pred.shape[1],1))
        return MyTensor.dot(losstensor,sumup_mask)