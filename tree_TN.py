import numpy as np
from numpy import random
from numpy import tensordot
from numpy import transpose
import matplotlib.pyplot as plt
from collections import namedtuple

class tree_TN():
    """
    This is a library describing tree tensor network, including optimization
    """
    def __init__(self,Dbond,d,Dout):
        """

        :param Dbond: bond dimension
        :param d: physical dimension
        :return:
        """
        self.Nlayer=0
        self.Dbond=Dbond
        self.d=d
        self.Dout=Dout

    def initialize(self,N):
        """

        :param nametuple: physical dimension, bond dimension
        :return: initial weight tensor
        """
        self.Nlayer=np.log2(N)-1
        W=[[] for i in range(self.Nlayer)]
        W[0]=[random.rand(self.d,self.d,self.d,self.d,self.Dbond) for i in range(N//4)]
        for i in range(1,self.Nlayer-1):
            W[i]=[random.rand(self.Dbond,self.Dbond,self.Dbond) for i in range(N//(2**(i+2)))]
        W[self.Nlayer]=[random.rand(self.Dbond,self.Dbond,self.Dout)]
        self.W=W

    def isometrize(self):
        """
        To make sure the weight tensor satisfy the isometric conditions, we apply QR decomposition on this weight
        :return:
        """
        for idx,w0 in enumerate(self.W[0]):
            temp=np.reshape(w0,[self.d**4,self.Dbond])
            dmin=min(temp.shape)
            Q,R=np.linalg.qr(temp)
            self.W[0][idx]=np.reshape(Q,[self.d,self.d,self.d,self.d,dmin])

        for i in range(1,self.Nlayer):
            for idx,wj in self.W[i]:
                temp=np.reshape(wj,[self.Dbond*self.Dbond,wj.shape[2]])
                Q,R=np.linalg.qr(temp)
                self.W[0][idx]=np.reshape(Q,[self.Dbond,self.Dbond,wj.shape[2]])



    def contractmpo(self,MPO):
        """

        :param MPO:
        :return:
        """
        n0=len(self.W[0])
        mpo_contracted=[]
        for i in range(n0):
            j=4*i
            while j<4*(i+1):
                mpo_i=tensordot(MPO[j],MPO[j+1],axes=(-1,2))
            mpo_i=transpose(mpo_i,[0,1,3,4,5,6,7,8,2,9])
            mpo_contracted.append(mpo_i)
        return mpo_contracted


    def conjugatepart(self,mpo_contracted):
        pass

    def environment(self,nlayer,m,mpo_contracted):
        """
        Calculate the environment of the nlayerth
        :param nlayer: the nth layer
        :param m: the mth tensor of nth layer
        :return: environment
        """


        if nlayer!=0:
            W0=[tensordot(tensordot(w0,mpo0,axes=([0,1,2,3],[0,2,4,6])),w0,axes=([2,3,4,5],[0,1,2,3]))\
                    for w0,mpo0 in zip(self.W[0],mpo_contracted)]
            W0=[transpose(w,[0,3,1,2]) for w in W0]
            f0=np.array([1])
            for w in W0:
                f0=tensordot(f0,w,axes=(-1,2))
                pass


            for i in range(1,nlayer):
                pass





    def update(self,nlayer,m,mpo_contracted):

        gamma=self.environment(nlayer,m,mpo_contracted)
        l=gamma.shape
        gamma1=gamma.reshape(np.prod(l[0:-1]),l[-1])
        dmin=min(gamma1)
        u,s,v=np.linalg.svd(gamma1)
        self.W[nlayer][m]=tensordot(u,s)
        pass


    def sweep(self,MPO):
        pass

