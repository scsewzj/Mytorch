import MyTensor
import MyPara
import numpy as np
from Gradlib import grad_basic,grad_connect

mytensor = MyTensor.mytensor

class loss:
    def __init__(self, comment=None):
        self.comment = comment
        self.scalar = 0
    def _loss(self, pred, true):
        pass
    def __call__(self, pred=None, true=None):
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

    ### mode : 'eco' or 'force'
    def backward(self,mode='eco'):
        G = mytensor.computegraph
        if mode=='eco':
            for i in G._lossnodes_index:
                G._nodelist[i]._grad = G._nodelist[i].ones()
            for i in G._parameternodes_index:
                G.grad_recursive(G._nodelist[i])
        if mode=='force':
            for i in G._nodelist:
                if type(i)==MyTensor.mytensor or type(i)==MyPara.myparameter:
                    if i.with_grad and i._grad is None:
                        i._grad = MyTensor.zeros(i.shape)
            self._tensorloss._grad = self._tensorloss.ones()
            stack = []
            passed = []
            for i in self._tensorloss._cg_ascend:
                if G._nodelist[i].with_grad:
                    stack.append((i,G._nodelist.index(self._tensorloss)))
            while stack != []:
                for i in stack:
                    edge = G._computegraph__find_edges(i[0],i[1])
                    if set(G._nodelist[i[1]]._grad_descend) == set(G._nodelist[i[1]]._cg_descend):
                        break
                #print(i)
                stack.remove(i)
                if i[0] not in passed:
                    for j in G._nodelist[i[0]]._cg_ascend:
                        if G._nodelist[j].with_grad:
                            stack.append((j,i[0]))
                    passed.append(i[0])
                #edge = G._computegraph__find_edges(edge[0],edge[1])
                if 'with' not in edge['forward'].keys():
                    local_grad = grad_basic(edge['forward']['op'],G._nodelist[edge['from']])
                else:
                    local_grad = grad_basic(edge['forward']['op'],G._nodelist[edge['from']],G._nodelist[edge['forward']['with']])
                if 'pos' in edge['forward'].keys():
                    G._nodelist[edge['from']]._grad = MyTensor.add.__wrapped__(grad_connect(edge['forward']['op'],local_grad,G._nodelist[edge['to']]._grad,edge['forward']['pos']),G._nodelist[edge['from']]._grad)
                else:
                    G._nodelist[edge['from']]._grad = MyTensor.add.__wrapped__(grad_connect(edge['forward']['op'],local_grad,G._nodelist[edge['to']]._grad),G._nodelist[edge['from']]._grad)
                G._nodelist[edge['from']]._grad_descend.append(edge['to'])
            

class L1(loss):
    def __init__(self, comment=None):
        super().__init__(comment)

    def _loss(self, pred: mytensor, true: mytensor):
        if len(pred.shape)!=2 or len(true.shape)!=2:
            raise ValueError('Only Support Dim: Batchsize * out_features, transfer to one-hot code or expand input data to 2-D (including batchsize)')
            return False
        losstensor = (MyTensor.add(pred,-1*true) ** 2) ** 0.5
        sumup_mask = MyTensor.ones((pred.shape[1],1))
        return MyTensor.dot(MyTensor.ones((1,pred.shape[0])),MyTensor.dot(losstensor,sumup_mask))/pred.shape[0]

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
    
class CrossEntropy(loss):
    def __init__(self, comment=None):
        super().__init__(comment)

    def _loss(self, pred: mytensor, true: mytensor):
        if len(pred.shape)!=2 or len(true.shape)!=2:
            raise ValueError('Only Support Dim: Batchsize * out_features, transfer to one-hot code or expand input data to 2-D (including batchsize)')
            return False
        log_pred = (pred+1e-8).log()
        losstensor=MyTensor.hadamard(true,log_pred)
        sumup_mask = MyTensor.ones((pred.shape[1],1))
        losstensor = (-1)*MyTensor.dot(losstensor,sumup_mask)
        return MyTensor.dot(MyTensor.ones((1,pred.shape[0])),losstensor)/pred.shape[0]
