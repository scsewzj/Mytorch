import random
import MyTensor
import pandas as pd
import numpy as np

def dataloader(datasource, batch_size, random_state = True, tensortype = 'tensor32'):
    index_list = []
    if len(datasource)<batch_size:
        raise ValueError('batch_size exceed the size of datasource')
    while len(index_list)<batch_size:
        i = random.randint(0,len(datasource)-1)
        if i not in index_list:
            index_list.append(i)
    data_label = np.array([datasource[i] for i in index_list])
    data = data_label[:,:-1]
    label = data_label[:,-1]
    return MyTensor.mytensor(data, tensortype=tensortype, with_grad=False), label



