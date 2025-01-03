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
            self.parameterb = myparameter(1,out_feature)

    def set_parameterw(self, *args):
        self.parameterw=myparameter(args, tensortype=self.tensortype)

    def set_parameterb(self, *args):
        self.parameterb=myparameter(args, tensortype=self.tensortype)

    def __call__(self, X:mytensor):
        return self.forward(X)

    def forward(self, X:mytensor):
        self._input = X
        self._forward = MyTensor.add(MyTensor.dot(self._input, self.parameterw),MyTensor.dot(MyTensor.ones((X.shape[0],1)),self.parameterb))
        return self._forward

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
