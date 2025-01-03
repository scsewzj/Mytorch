import numpy as np
from ComputeGraph import computegraph
#import util




class mytensor():

    computegraph = computegraph()

    def __init__(self, npar_data, grad=None, tensortype='tensor32', with_grad=False):
        
        self.tensortype=tensortype
        self.with_grad=with_grad
        
        ### heritage implement later
        if type(npar_data)==np.array:
            self.npar_data=npar_data
        else:
            if tensortype == 'tensor64':
                self.npar_data = np.array(npar_data, dtype='float64')
            else:
                self.npar_data = np.array(npar_data, dtype='float32')
        
        if with_grad:
            if grad is None:
                self.grad = np.zeros(self.npar_data.shape)
            else:
                self.grad = grad
        else:
            self.grad = None

        self.__get_shape()

    ### call the class to initialize a tensor
    def __call__(self):
        pass
    
    def __str__(self):
        return self.tensortype+'('+str(self.npar_data)+')'

    def __repr__(self):
        return self.tensortype+'('+str(self.npar_data)+')'

    ### subscriptable
    def __getitem__(self,index):
        return mytensor(self.npar_data.__getitem__(index),grad=self.grad.__getitem__(index),tensortype=self.tensortype,with_grad=self.with_grad)

    def T(self,grad=None):
        return mytensor(self.npar_data.T,grad=None,tensortype=self.tensortype,with_grad=self.with_grad)
        ### default: grad will be cleared
        ### but provide the para grad which could be manually set

    ### temporarily stop recording grad
    def pause_grad(self):
        self.with_grad=False
    
    ### recover the grad from temporary pause
    def recover_grad(self):
        self.with_grad=True
    
    ### turn off recoring grad permenently which will clear the existed grad
    def off_grad(self):
        self.with_grad=False
        self.grad=None

    ### reinitialize grad in the case of offed grad or not initialized grad
    def reinit_grad(self,grad=None):
        self.with_grad=True
        if grad is None:
            self.grad=np.zeros(self.npar_data.shape)
        else:
            self.grad=grad


    def ones(self):
        return mytensor(np.ones(self.npar_data.shape), tensortype=self.tensortype)

    def zeros(self):
        return mytensor(np.zeros(self.npar_data.shape), tensortype=self.tensortype)

    def __get_shape(self):
        self.shape = self.npar_data.shape

    @computegraph.__record_tensor_oper__
    def __mul__(self, val):
        return mytensor(self.npar_data*val, tensortype=self.tensortype)

    @computegraph.__record_tensor_oper__
    def __rmul__(self, val):
        return mytensor(self.npar_data*val, tensortype=self.tensortype)

    @computegraph.__record_tensor_oper__
    def __add__(self, val):
        return mytensor(self.npar_data+val, tensortype=self.tensortype)    

    @computegraph.__record_tensor_oper__
    def __radd__(self, val):
        return mytensor(self.npar_data+val, tensortype=self.tensortype)

    @computegraph.__record_tensor_oper__
    def __sub__(self, val):
        return mytensor(self.npar_data-val, tensortype=self.tensortype)

    @computegraph.__record_tensor_oper__
    def __rsub__(self, val):
        return mytensor(val - self.npar_data, tensortype=self.tensortype)

    @computegraph.__record_tensor_oper__
    def __truediv__(self, val):
        return mytensor(self.npar_data/val, tensortype=self.tensortype)

    @computegraph.__record_tensor_oper__
    def __rtruediv__(self, val):
        return mytensor(val/self.npar_data, tensortype=self.tensortype)

    @computegraph.__record_tensor_oper__
    def __pow__(self, val):
        return mytensor(self.npar_data ** val, tensortype=self.tensortype)

@mytensor.computegraph.__record_tensor_oper__
def dot(tensor1: mytensor, tensor2: mytensor):
    return mytensor(np.dot(tensor1.npar_data,tensor2.npar_data), tensortype=tensor1.tensortype)

@mytensor.computegraph.__record_tensor_oper__
def add(tensor1: mytensor, tensor2: mytensor):
    return mytensor(tensor1.npar_data+tensor2.npar_data, tensortype=tensor1.tensortype)

@mytensor.computegraph.__record_tensor_oper__
def hadamard(tensor1: mytensor, tensor2: mytensor):
    return mytensor(tensor1.npar_data*tensor2.npar_data, tensortype=tensor1.tensortype)

def zeros(*args, tensortype='tensor32'):
    return mytensor(np.zeros(*args), tensortype=tensortype)

def ones(*args, tensortype='tensor32'):
    return mytensor(np.ones(*args), tensortype=tensortype)

def random(*args, tensortype='tensor32'):
    return mytensor(np.random.randn(*args), tensortype=tensortype)

