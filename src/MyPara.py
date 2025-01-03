import MyTensor
import numpy as np


class myparameter(MyTensor.mytensor):

    def __init__(self, *args, npar_data=None, tensortype='tensor32'):
        if npar_data is None:
            npar_data = np.random.randn(*args)/np.sqrt(args[0])
        super().__init__(npar_data, tensortype=tensortype)
        self.with_grad = True
        self._is_para = True

        

    def __update(self,lr):
        self.npar_data=self.npar_data-(self._grad.npar_data * lr)
    
    def __str__(self):
        return str(type(self)).split('\'')[1].split('.')[1]+'('+str(self.npar_data)+', '+self.tensortype+')'

    def __repr__(self):
        return str(type(self)).split('\'')[1].split('.')[1]+'('+str(self.npar_data)+', '+self.tensortype+')'

