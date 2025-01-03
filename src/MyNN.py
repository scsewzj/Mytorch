import MyTensor
from MyPara import *
import numpy as np
from ComputeGraph import *
from MyModel import *

mytensor = MyTensor.mytensor

class my_linear_layer:

    def __init__(self, in_feature, out_feature, paraw=None, parab=None, tensortype='tensor32', alias = None):
        self.tensortype = tensortype
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.alias = alias
        if paraw is not None:
            self.parameterw = paraw
        else:
            self.parameterw = myparameter(in_feature,out_feature)
        if parab is not None:
            self.parameterb = parab 
        else:
            self.parameterb = myparameter(npar_data=np.zeros((1,out_feature)), tensortype=tensortype)

    #def set_parameterw(self, *args):
    #    self.parameterw=myparameter(args, tensortype=self.tensortype)

    #def set_parameterb(self, *args):
    #    self.parameterb=myparameter(args, tensortype=self.tensortype)

    def __call__(self, X:mytensor):
        return self.forward(X)

    def forward(self, X:mytensor):
        return MyTensor.add(MyTensor.dot(X, self.parameterw),MyTensor.dot(MyTensor.ones((X.shape[0],1)),self.parameterb))

    def _parametercursor_CG(self):
        G = mytensor.computegraph
        for i in G._nodelist:
            if i==self._tensorloss and G._nodelist.index(i) not in G._parameternodes_index:
                G._parameternodes_index(G._nodelist.index(i))
    
    def __str__(self):
        if self.alias is None:
            return str(type(self)).split('\'')[1].split('.')[1]+'(\nw:'+str(self.parameterw)+',\nb:'+str(self.parameterb)+'\nin_feature:'+str(self.in_feature) + ', out_feature:'+str(self.out_feature)+')'
        else:
            return str(type(self)).split('\'')[1].split('.')[1]+'('+self.alias+')'+'(\nw:'+str(self.parameterw)+',\nb:'+str(self.parameterb)+'\nin_feature:'+str(self.in_feature) + ', out_feature:'+str(self.out_feature)+')'

    def __repr__(self):
        if self.alias is None:
            return str(type(self)).split('\'')[1].split('.')[1]+'(\nw:'+str(self.parameterw)+',\nb:'+str(self.parameterb)+'\nin_feature:'+str(self.in_feature) + ', out_feature:'+str(self.out_feature)+')'
        else:
            return str(type(self)).split('\'')[1].split('.')[1]+'('+self.alias+')'+'(\nw:'+str(self.parameterw)+',\nb:'+str(self.parameterb)+'\nin_feature:'+str(self.in_feature) + ', out_feature:'+str(self.out_feature)+')'


class my_relu:

    def __init__(self, alias = None):
        self.alias = alias

    def __call__(self, X:mytensor):
        return self.forward(X)

    def forward(self, X:mytensor):
        mask = X.npar_data>=0
        mask = mytensor(mask.astype('float32'))
        return MyTensor.hadamard(X,mask)

    def __str__(self):
        if self.alias is None:
            return str(type(self)).split('\'')[1].split('.')[1]+'()'
        else:
            return str(type(self)).split('\'')[1].split('.')[1]+'('+self.alias+')'+'()'

    def __repr__(self):
        if self.alias is None:
            return str(type(self)).split('\'')[1].split('.')[1]+'()'
        else:
            return str(type(self)).split('\'')[1].split('.')[1]+'('+self.alias+')'+'()'

class my_softmax:
    def __init__(self, alias = None):
        self.alias = alias

    def __call__(self, X:mytensor):
        return self.forward(X)

    def forward(self, X:mytensor):
        temp = X.exp()
        tempsum = MyTensor.dot(temp,MyTensor.ones((X.shape[1],1)))
        tempsum_sameshape = MyTensor.dot(tempsum, MyTensor.ones((1, X.shape[1])))
        rev_tempsum_sameshape = tempsum_sameshape**(-1)
        return MyTensor.hadamard(X.exp(),rev_tempsum_sameshape)

    def __str__(self):
        if self.alias is None:
            return str(type(self)).split('\'')[1].split('.')[1]+'()'
        else:
            return str(type(self)).split('\'')[1].split('.')[1]+'('+self.alias+')'+'()'

    def __repr__(self):
        if self.alias is None:
            return str(type(self)).split('\'')[1].split('.')[1]+'()'
        else:
            return str(type(self)).split('\'')[1].split('.')[1]+'('+self.alias+')'+'()'

class my_softmax_stable:
    def __init__(self, alias = None):
        self.alias = alias

    def __call__(self, X:mytensor):
        return self.forward(X)

    def forward(self, X:mytensor):
        maxmask = []
        for i in X.npar_data:
            maxmask.append(i==np.max(i).astype(int))
        XMax=MyTensor.hadamard(X,MyTensor.mytensor(np.array(maxmask)))
        XMax = MyTensor.dot(XMax,MyTensor.ones((X.shape[1],1)))
        XMax = MyTensor.dot(XMax, MyTensor.ones((1, X.shape[1])))
        temp = MyTensor.add(X,(-1)*XMax).exp()
        tempsum = MyTensor.dot(temp,MyTensor.ones((X.shape[1],1)))
        tempsum_sameshape = MyTensor.dot(tempsum, MyTensor.ones((1, X.shape[1])))
        return MyTensor.hadamard(temp,1/tempsum_sameshape)

    def __str__(self):
        if self.alias is None:
            return str(type(self)).split('\'')[1].split('.')[1]+'()'
        else:
            return str(type(self)).split('\'')[1].split('.')[1]+'('+self.alias+')'+'()'

    def __repr__(self):
        if self.alias is None:
            return str(type(self)).split('\'')[1].split('.')[1]+'()'
        else:
            return str(type(self)).split('\'')[1].split('.')[1]+'('+self.alias+')'+'()'

class my_1dcov:
    pass

class my_2dcov:
    pass

class my_maxpool:
    pass

class my_flatten:
    pass

class my_dropout:
    pass

