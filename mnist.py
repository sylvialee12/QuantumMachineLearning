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


@functimer
def data_process2D(mnist,margin,pool,kernel):
    reshaped_tar=mnist.target.reshape([len(mnist.target),1])
    Data=np.concatenate((mnist.data, reshaped_tar), axis=1)
    shuffle(Data)
    data0=Data[:,0:mnist.data.shape[1]]
    target=Data[:,-1]

    psi0=data0/255.0

    pixel=int(np.sqrt(data0.shape[1]))
    psi3=psi0.reshape([data0.shape[0],pixel,pixel])
    margincol=np.zeros([data0.shape[0],pixel,margin])
    psi3=np.concatenate((margincol,psi3,margincol),axis=2)
    marginrow=np.zeros([data0.shape[0],margin,psi3.shape[2]])
    psi3=np.concatenate((marginrow,psi3,marginrow),axis=1)
    if kernel=="cosine":
        psi1=np.cos(np.pi/2*psi3).reshape(psi3.shape+(1,))
        psi2=np.sin(np.pi/2*psi3).reshape(psi3.shape+(1,))
    elif kernel=="linear":
        psi1=psi3.reshape(psi3.shape+(1,))
        psi2=1.0/4*psi3.reshape(psi3.shape+(1,))
    data=np.concatenate((psi1,psi2),axis=-1)
    lvector=np.zeros([target.shape[0],10])
    for (i,l) in enumerate(lvector):
        l[int(target[i])]=1
    if pool==0 or pool==1:
        return data,lvector
    else:
        u=np.kron(np.eye((pixel+2*margin)//pool),1/pool*np.ones([pool,1]))
        psi3_pooled=np.tensordot(psi3,u,axes=(2,0))
        psi3_pooled=np.tensordot(np.transpose(u),psi3_pooled,axes=(1,1))
        psi3_pooled=np.transpose(psi3_pooled,[1,0,2])
        if kernel=="cosine":
            psi1_pooled=np.cos(np.pi/2*psi3_pooled).reshape(psi3_pooled.shape+(1,))
            psi2_pooled=np.sin(np.pi/2*psi3_pooled).reshape(psi3_pooled.shape+(1,))
        elif kernel=="linear":
            psi1_pooled=psi3_pooled.reshape(psi3_pooled.shape+(1,))
            psi2_pooled=1.0/4*psi3_pooled.reshape(psi3_pooled.shape+(1,))
        data=np.concatenate((psi1_pooled,psi2_pooled),axis=-1)
        return data,lvector

def data_process2DwithSVD(mnist,margin,pool):
    reshaped_tar=mnist.target.reshape([len(mnist.target),1])
    Data=np.concatenate((mnist.data, reshaped_tar), axis=1)
    shuffle(Data)
    data0=Data[:,0:mnist.data.shape[1]]
    target=Data[:,-1]

    psi0=data0/255.0

    pixel=int(np.sqrt(data0.shape[1]))
    psi3=psi0.reshape([data0.shape[0],pixel,pixel])
    margincol=np.zeros([data0.shape[0],pixel,margin])
    psi3=np.concatenate((margincol,psi3,margincol),axis=2)
    marginrow=np.zeros([data0.shape[0],margin,psi3.shape[2]])
    psi3=np.concatenate((marginrow,psi3,marginrow),axis=1)
    lvector=np.zeros([target.shape[0],10])
    for (i,l) in enumerate(lvector):
        l[int(target[i])]=1

    u,s,v=np.linalg.svd(psi3)
    if pool is None or pool==0 or pool==1:
        compressed_pixel=(pixel+2*margin)//1
    else:
        compressed_pixel=(pixel+2*margin)//pool
    psi_pooled=np.zeros([data0.shape[0],compressed_pixel,compressed_pixel])
    for idx,psi3_i in enumerate(psi3):
        psi_pooled[idx,:,:]=np.dot(u[idx,:compressed_pixel,:compressed_pixel],np.dot(np.diag(s[idx,:compressed_pixel]),v[idx,:compressed_pixel,:compressed_pixel]))
    psi1_pooled=np.cos(np.pi/2*psi_pooled).reshape(psi_pooled.shape+(1,))
    psi2_pooled=np.sin(np.pi/2*psi_pooled).reshape(psi_pooled.shape+(1,))
    data=np.concatenate((psi1_pooled,psi2_pooled),axis=-1)
    return data,lvector




if __name__=="__main__":
    print("yes")







