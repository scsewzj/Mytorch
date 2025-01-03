import MyTensor
import numpy as np
import math
import scipy as sp
import Gradlib
import functools
import copy

class computegraph:

    def __init__(self, nodelist = [], edgelist = [], super_graph=None):
        if super_graph is not None:
            self._nodelist = super_graph.nodelist
            self._edgelist = super_graph.edgelist
        else:
            self._nodelist = nodelist
            self._edgelist = edgelist
        self._lossnodes_index = []
        self._parameternodes_index = []
        self._parameteredges_index = []
        self._flow_reversed = []
        self._valid_path_reversed = []

    def __recordnode(self, node):
        if node not in self._nodelist:
            self._nodelist.append(node)

    def __recordedge(self, edge):
        if edge not in self._edgelist:
            self._edgelist.append(edge)

    def __record_tensor_oper__(self,func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result=func(*args, **kwargs)
            if args[0] not in self._nodelist:
                self.__recordnode(args[0])
                node1 = len(self._nodelist)-1
            else:
                node1 = self._nodelist.index(args[0])
            if args[1] not in self._nodelist:
                self.__recordnode(args[1])
                node2 = len(self._nodelist)-1
            else:
                node2 = self._nodelist.index(args[1])
            if result not in self._nodelist:
                self.__recordnode(result)
                node_result = len(self._nodelist)-1
            else:
                node_result = self._nodelist.index(result)

            flag_para = False
            if str(type(self._nodelist[node1]))=="<class 'MyPara.myparameter'>":
                self._parameternodes_index.append(node1)
                flag_para = True
            if str(type(self._nodelist[node2]))=="<class 'MyPara.myparameter'>":
                self._parameternodes_index.append(node2)
                flag_para = True
            

            tensor_or_para = ["<class 'MyTensor.mytensor'>", "<class 'MyPara.myparameter'>"]
            if str(type(args[0])) in tensor_or_para and str(type(args[1])) in tensor_or_para:
                self.__recordedge({'from': node1, 'to': node_result, 'forward': {'op':str(func).split()[1],'with':node2,'pos':'right'}})
                self.__recordedge({'from': node2, 'to': node_result, 'forward': {'op':str(func).split()[1],'with':node1}, 'pos':'left'})
                if flag_para:
                    self._parameteredges_index.append(len(self._edgelist)-1)
                    self._parameteredges_index.append(len(self._edgelist)-2)
            else:
                self.__recordedge({'from': node1, 'to': node_result, 'forward': {'op':str(func).split()[1],'with':node2}})
                if flag_para:
                    self._parameteredges_index.append(len(self._edgelist)-1)
            return result
        return wrapper

    def __find_edges_via_nodeto(self, node):
        potential_edges = []
        for i in self._edgelist:
            if i['to']==node:
                potential_edges.append(i)
        return potential_edges


    def __backtrace_guidegraph__(self):
        """
        def copy_dict(oridict):
            new_dict={}
            for i in oridict.keys():
                new_dict[i]=oridict[i]
            return new_dict
        """

        for i in self._lossnodes_index:
            self._flow_reversed.append({'loss':i , 'pointer':i, 'path' : [] , 'flag' : False})

        while len(self._flow_reversed)>0:
            op = self._flow_reversed[0]
            potential_edges = self.__find_edges_via_nodeto(op['pointer'])
            if len(potential_edges)==0:
                op['flag']=True
                self._valid_path_reversed.append(op)
            else:
                for i in potential_edges:
                    loss = op['loss']
                    pointer=i['from']
                    path = copy.deepcopy(op['path'])
                    path.append(i)
                    self._flow_reversed.append({'loss':loss , 'pointer':pointer, 'path' : path, 'flag' : False})
            self._flow_reversed.remove(self._flow_reversed[0])
    

    def _path_filter(self):
        items = []
        filtered = []
        for i in range(len(self._valid_path_reversed)):
            for j in self._valid_path_reversed[i]['path']:
                items.append(j['from'])
                items.append(j['forward']['with'])
            if len(set(self._parameternodes_index)&set(items))>0:
                filtered.append(i)
            items = []
        temp = [self._valid_path_reversed[i] for i in filtered]
        self._valid_path_reversed = temp

    def __autograd__(self, L=None):
        tensor_or_para = ["<class 'MyTensor.mytensor'>", "<class 'MyPara.myparameter'>"]
        for i in self._valid_path_reversed:
            for j in i['path']:
                j['grad']=Gradlib.grad_basic(operation=j['forward']['op'],operant=j['from'],with_operant=j['forward']['with'])
                #if str(type(j['forward']['with'])) in tensor_or_para:
                #    tensorgrad = L._tensorloss
                #    pos = i['path'].index(j)
                #    for k in range(pos+1):
                        ### continuous grad?
                        #if i['path'][k]['forward']['op']==
                        #tensorgrad=MyTensor.dot.__wrapped__(tensorgrad, i['path'][k]['grad'])
                #    print(tensorgrad)