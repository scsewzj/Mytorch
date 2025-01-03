import MyTensor


class gradlib:
    def __grad_basic_dot(tensor, with_tensor):
        return with_tensor.T()

    def __grad_basic_add(tensor, with_tensor):
        return tensor.ones()

    def __grad_basic_hadamard(tensor, with_tensor):
        return with_tensor

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
        temp = MyTensor.hadamard.__wrapped__(tensor.__rtruediv__.__wrapped__(tensor,1),tensor.__rtruediv__.__wrapped__(tensor,1))
        return temp.__mul__.__wrapped__(temp,-val)

    def __grad_basic_mytensor_pow(tensor, val):
       return  tensor.__pow__.__wrapped__(tensor,val-1).__mul__.__wrapped__(tensor.__pow__.__wrapped__(tensor,val-1),val)



    _basic_op = [
        'dot',
        'add',
        'hadamard',
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

    _grad_basic_funcs = [
        __grad_basic_dot,
        __grad_basic_add,
        __grad_basic_hadamard,
        __grad_basic_mytensor_add,
        __grad_basic_mytensor_radd,
        __grad_basic_mytensor_sub,
        __grad_basic_mytensor_rsub,
        __grad_basic_mytensor_mul,
        __grad_basic_mytensor_rmul,
        __grad_basic_mytensor_truediv,
        __grad_basic_mytensor_rtruediv,
        __grad_basic_mytensor_pow
    ]

    '''
    _connecting_funcs = {
        # dot
        'dot': #__grad_basic_dot,


        # add
        'add': #__grad_basic_add,
        
        # hadamard
        'hadamard': #__grad_basic_hadamard,


        # hadamard
        'mytensor.__add__': #__grad_basic_mytensor_add,
        'mytensor.__radd__': #__grad_basic_mytensor_radd,
        'mytensor.__sub__': #__grad_basic_mytensor_sub,
        'mytensor.__rsub__': #__grad_basic_mytensor_rsub,
        'mytensor.__mul__': #__grad_basic_mytensor_mul,
        'mytensor.__rmul__': #__grad_basic_mytensor_rmul,
        'mytensor.__truediv__': #hadamard
        'mytensor.__rtruediv__': #hadamard
        'mytensor.__pow__': # hadamard
    }
    '''

def grad_basic(operation, operant, with_operant):
    if operation not in gradlib._basic_op:
        raise TypeError('The operation type is not supported to be differenciated')
        return -1
    else:
        return gradlib._grad_basic_funcs[gradlib._basic_op.index(operation)](operant, with_operant)


