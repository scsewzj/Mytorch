import MyTensor
import numpy as np


class myparameter(MyTensor.mytensor):

    def __init__(self, *args, npar_data=None, tensortype='tensor32'):
        super().__init__(npar_data, tensortype=tensortype)
        self.with_grad = True
        self._is_para = True

        if npar_data is None:
            self.npar_data = np.random.randn(*args)

    def __update(self,lr):
        self.npar_data+=self.grad * lr

    
    def __str__(self):
        return str(type(self)).split('\'')[1].split('.')[1]+'('+str(self.npar_data)+', '+self.tensortype+')'

    def __repr__(self):
        return str(type(self)).split('\'')[1].split('.')[1]+'('+str(self.npar_data)+', '+self.tensortype+')'

