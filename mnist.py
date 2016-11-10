"""
This is to load the mnist data and process data
"""

import numpy as np
from sklearn.datasets import fetch_mldata
from time import time
from numpy.random import shuffle

def functimer(func):
    def wrapper(*args,**kwargs):
        now=time()
        num=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time()-now))
        return num
    return wrapper

def load_data():
    mnist=fetch_mldata('MNIST original')
    return mnist

@functimer
def data_process(mnist,n):

    reshaped_tar=mnist.target.reshape([len(mnist.target),1])
    Data=np.concatenate((mnist.data, reshaped_tar), axis=1)
    shuffle(Data)
    data0=Data[:,0:mnist.data.shape[1]]
    target=Data[:,-1]
    # psi1=data0/(255.0)
    # psi2=np.sqrt(1-psi1**2)
    psi0=data0/255.0
    psi1=np.cos(np.pi/2*psi0)
    psi2=np.sin(np.pi/2*psi0)
    data=np.vstack([psi1,psi2])
    data=data.reshape([2,data0.shape[0],data0.shape[1]])
    data=np.transpose(data,[1,2,0])

    lvector=np.zeros([target.shape[0],10])
    for (i,l) in enumerate(lvector):
        l[int(target[i])]=1
    if n==0 or n==1:
        return data,lvector
    else:
        pixel=int(np.sqrt(data0.shape[1]))
        psi3=psi0.reshape([data0.shape[0],pixel,pixel])
        u=np.kron(np.eye(pixel//n),1/n*np.ones([n,1]))
        psi_pooled=np.tensordot(psi3,u,axes=(-1,0))
        psi_pooled=np.tensordot(np.transpose(u),psi_pooled,axes=(-1,1))
        psi_pooled=np.transpose(psi_pooled,[1,0,2])
        psi_pooled=psi_pooled.reshape([data0.shape[0],pixel//n*(pixel//n)])
        psi1_pooled=np.cos(np.pi/2*psi_pooled)
        psi2_pooled=np.sin(np.pi/2*psi_pooled)
        data_pooled=np.vstack([psi1_pooled,psi2_pooled])
        data_pooled=data_pooled.reshape([2,psi_pooled.shape[0],psi_pooled.shape[1]])
        data_pooled=np.transpose(data_pooled,[1,2,0])

        return data_pooled,lvector









