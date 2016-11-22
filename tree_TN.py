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
        self.Nlayer=np.log2(N)
        W=[[] for i in range(self.Nlayer)]
        W[0]=[random.rand(self.d,self.d,self.Dbond) for i in range(N//2)]
        for i in range(1,self.Nlayer-1):
            W[i]=[random.rand(self.Dbond,self.Dbond,self.Dbond) for i in range(N//(2**(i+1)))]
        W[self.Nlayer]=[random.rand(self.Dbond,self.Dbond,self.Dout)]
        self.W=W

    def isometrize(self):
        """
        To make sure the weight tensor satisfy the isometric conditions, we apply QR decomposition on this weight
        :return:
        """
        for idx,w0 in enumerate(self.W[0]):
            temp=np.reshape(w0,[self.d**2,self.Dbond])
            dmin=min(temp.shape)
            Q,R=np.linalg.qr(temp)
            self.W[0][idx]=np.reshape(Q,[self.d,self.d,dmin])

        for i in range(1,self.Nlayer):
            for idx,wj in self.W[i]:
                temp=np.reshape(wj,[self.Dbond*self.Dbond,wj.shape[2]])
                Q,R=np.linalg.qr(temp)
                self.W[0][idx]=np.reshape(Q,[self.Dbond,self.Dbond,wj.shape[2]])


    def contract_full(self,left_tensor,right_tensor,down_tensor,up_tensor):
        """
        :param left_tensor:
        :param right_tensor:
        :param up_tensor:
        :param down_tensor:
        :return:
        """
        #
        # 
        if len(left_tensor.shape)==len(right_tensor.shape) and up_tensor is not None:
            temp=tensordot(left_tensor,right_tensor,axes=(3,2))
            temp=tensordot(up_tensor,temp,axes=([0,1,[0,3]]))
            temp=tensordot(temp,down_tensor,axes=([1,3],[0,1]))
            output=temp.reshape(0,3,1,2)
        elif up_tensor is None:
            temp=tensordot(left_tensor,right_tensor,axes=(3,2))
            temp=tensordot(temp,down_tensor,axes=([1,4],[0,1]))
            output=temp.reshape(0,2,4,1,3)
        elif len(left_tensor)==5:
            temp=tensordot(left_tensor,right_tensor,axes=(4,2))
            temp=tensordot(up_tensor,temp,axes=(1,4))
            temp=tensordot(temp,down_tensor,axes=([4,6],[0,1]))
            output=temp.reshape(1,6,4,5,0,2,3)
        elif len(right_tensor)==5:
            temp=tensordot(left_tensor,right_tensor,axes=(3,3))
            temp=tensordot(up_tensor,temp,axes=(0,0))
            temp=tensordot(temp,down_tensor,axes=([2,6],[0,1]))
            output=temp.reshape(1,6,2,5,0,3,4)
        elif len(left_tensor)==7:
            temp=tensordot(left_tensor,right_tensor,axes=(3,2))
            temp=tensordot(up_tensor,temp,axes=([0,1],[0,6]))
            temp=tensordot(temp,down_tensor,axes=([1,6],[0,1]))
            output=temp.reshape(0,6,1,5,2,3,4)
        elif len(right_tensor)==7:
            temp=tensordot(left_tensor,right_tensor,axes=(3,2))
            temp=tensordot(up_tensor,temp,axes=([0,1],[0,3]))
            temp=tensordot(temp,down_tensor,axes=([1,3],[0,1]))
            output=temp.reshape(0,6,1,2,3,4,5)
        return output



    def environment(self,nlayer,m,mpo):
        """
        Calculate the environment of the nlayerth
        :param nlayer: the nth layer
        :param m: the mth tensor of nth layer
        :return: environment
        """
        for i in range(self.Nlayer):
            mpo_i=[]
            for j in range(len(mpo)//2):
                if i!=nlayer or j!=m:
                    mpo_i.append(self.contract_full(mpo[2*j],mpo[2*j+1],self.W[i][j],self.W[i][j]))
                else:
                    mpo_i.append(self.contract_full(mpo[2*j],mpo[2*j+1],self.W[i][j],None))
            mpo=mpo_i

        return np.squeeze(mpo)



    def update(self,nlayer,m,mpo):

        gamma=self.environment(nlayer,m,mpo)
        l=gamma.shape
        gamma1=gamma.reshape(np.prod(l[0:-1]),l[-1])
        dmin=min(gamma1)
        u,s,v=np.linalg.svd(gamma1)
        self.W[nlayer][m]=tensordot(u,s)
        pass


    def sweep(self,MPO):
        pass

