import MyTensor


class gradlib:
    def __grad_basic_dot(tensor, with_tensor):
        return with_tensor.T()

    def __grad_basic_add(tensor, with_tensor):
        return tensor.ones()

    def __grad_basic_hadamard(tensor, with_tensor):
        return with_tensor
    
    def __grad_basic_mytensor_exp(tensor):
        return tensor.exp.__wrapped__(tensor)
    
    def __grad_basic_mytensor_log(tensor):
        return tensor.__pow__.__wrapped__(tensor,-1)

    def __grad_basic_mytensor_add(tensor, val):
        return tensor.ones()

    def __grad_basic_mytensor_radd(tensor, val):
        return tensor.ones()

    def __grad_basic_mytensor_sub(tensor, val):
        return tensor.ones()

    def __grad_basic_mytensor_rsub(tensor, val):
        return tensor.ones().__rsub__.__wrapped__(tensor.ones(),0)

    def __grad_basic_mytensor_mul(tensor, val):
        return tensor.ones().__mul__.__wrapped__(tensor.ones(),val)

    def __grad_basic_mytensor_rmul(tensor, val):
        return tensor.ones().__mul__.__wrapped__(tensor.ones(),val)

    def __grad_basic_mytensor_truediv(tensor, val):
        return tensor.ones().__truediv__.__wrapped__(tensor.ones(),val)

    def __grad_basic_mytensor_rtruediv(tensor, val):
        return tensor.__pow__.__wrapped__(tensor,-2).__mul__.__wrapped__(tensor.__pow__.__wrapped__(tensor,-2),-val)

    def __grad_basic_mytensor_pow(tensor, val):
       return  tensor.__pow__.__wrapped__(tensor,val-1).__mul__.__wrapped__(tensor.__pow__.__wrapped__(tensor,val-1),val)


    """
    _basic_op = [
        'dot',
        'add',
        'hadamard',
        'exp',
        'log',
        'mytensor.__add__',
        'mytensor.__radd__',
        'mytensor.__sub__',
        'mytensor.__rsub__',
        'mytensor.__mul__',
        'mytensor.__rmul__',
        'mytensor.__truediv__',
        'mytensor.__rtruediv__',
        'mytensor.__pow__'
    ]
    """
    

    _grad_basic_funcs = {
        'dot':__grad_basic_dot,
        'add':__grad_basic_add,
        'hadamard':__grad_basic_hadamard,
        
        'mytensor.__add__':__grad_basic_mytensor_add,
        'mytensor.__radd__':__grad_basic_mytensor_radd,
        'mytensor.__sub__':__grad_basic_mytensor_sub,
        'mytensor.__rsub__':__grad_basic_mytensor_rsub,
        'mytensor.__mul__':__grad_basic_mytensor_mul,
        'mytensor.__rmul__':__grad_basic_mytensor_rmul,
        'mytensor.__truediv__':__grad_basic_mytensor_truediv,
        'mytensor.__rtruediv__':__grad_basic_mytensor_rtruediv,
        'mytensor.__pow__':__grad_basic_mytensor_pow,
        'mytensor.exp':__grad_basic_mytensor_exp,
        'mytensor.log':__grad_basic_mytensor_log
    }




def grad_basic(operation, operant, with_operant=None):
    if operation not in gradlib._grad_basic_funcs.keys():
        raise TypeError('The operation type is not supported to be differenciated')
        return -1
    else:
        if with_operant is None:
            return gradlib._grad_basic_funcs[operation](operant)
        else:
            return gradlib._grad_basic_funcs[operation](operant, with_operant)

def grad_connect(operation, local_grad, previous_grad, pos='right'):
    if operation == 'dot':
        if pos == 'right':
            return MyTensor.dot.__wrapped__(previous_grad, local_grad)
        else:
            return MyTensor.dot.__wrapped__(local_grad, previous_grad)
    else:
        return MyTensor.hadamard.__wrapped__(local_grad, previous_grad)

"""
    _connect_funcs = {
        'dot':MyTensor.dot.__wrapped__,
        'add':MyTensor.hadamard.__wrapped__,
        'hadamard':MyTensor.hadamard.__wrapped__,
        'mytensor.__add__':MyTensor.hadamard.__wrapped__,
        'mytensor.__radd__':MyTensor.hadamard.__wrapped__,
        'mytensor.__sub__':MyTensor.hadamard.__wrapped__,
        'mytensor.__rsub__':MyTensor.hadamard.__wrapped__,
        'mytensor.__mul__':MyTensor.hadamard.__wrapped__,
        'mytensor.__rmul__':MyTensor.hadamard.__wrapped__,
        'mytensor.__truediv__':MyTensor.hadamard.__wrapped__,
        'mytensor.__rtruediv__':MyTensor.hadamard.__wrapped__,
        'mytensor.__pow__':MyTensor.hadamard.__wrapped__    
    }
"""

