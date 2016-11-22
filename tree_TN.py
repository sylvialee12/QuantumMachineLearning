import numpy as np
from numpy import random
from numpy import tensordot
import MPOIsing
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
        self.Nlayer=np.log2(N).astype('int')
        W=[[] for i in range(self.Nlayer)]
        W[0]=[random.rand(self.d,self.d,self.Dbond) for i in range(N//2)]
        for i in range(1,self.Nlayer-1):
            W[i]=[random.rand(self.Dbond,self.Dbond,self.Dbond) for i in range(N//(2**(i+1)))]
        W[self.Nlayer-1]=[random.rand(self.Dbond,self.Dbond,self.Dout)]
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
            for idx,wj in enumerate(self.W[i]):
                temp=np.reshape(wj,[self.Dbond*self.Dbond,wj.shape[2]])
                Q,R=np.linalg.qr(temp)
                self.W[i][idx]=np.reshape(Q,[self.Dbond,self.Dbond,wj.shape[2]])


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
            temp=tensordot(up_tensor,temp,axes=([0,1],[0,3]))
            temp=tensordot(temp,down_tensor,axes=([1,3],[0,1]))
            output=temp.transpose(0,3,1,2)
        elif up_tensor is None:
            temp=tensordot(left_tensor,right_tensor,axes=(3,2))
            temp=tensordot(temp,down_tensor,axes=([1,4],[0,1]))
            output=temp.transpose(0,2,4,1,3)
        elif len(left_tensor.shape)==5:
            temp=tensordot(left_tensor,right_tensor,axes=(4,2))
            temp=tensordot(up_tensor,temp,axes=(1,4))
            temp=tensordot(temp,down_tensor,axes=([4,6],[0,1]))
            output=temp.transpose(1,6,4,5,0,2,3)
        elif len(right_tensor.shape)==5:
            temp=tensordot(left_tensor,right_tensor,axes=(3,3))
            temp=tensordot(up_tensor,temp,axes=(0,0))
            temp=tensordot(temp,down_tensor,axes=([2,6],[0,1]))
            output=temp.transpose(1,6,2,5,0,3,4)
        elif len(left_tensor.shape)==7:
            temp=tensordot(left_tensor,right_tensor,axes=(3,2))
            temp=tensordot(up_tensor,temp,axes=([0,1],[0,6]))
            temp=tensordot(temp,down_tensor,axes=([1,6],[0,1]))
            output=temp.transpose(0,6,1,5,2,3,4)
        elif len(right_tensor.shape)==7:
            temp=tensordot(left_tensor,right_tensor,axes=(3,2))
            temp=tensordot(up_tensor,temp,axes=([0,1],[0,3]))
            temp=tensordot(temp,down_tensor,axes=([1,3],[0,1]))
            output=temp.transpose(0,6,1,2,3,4,5)
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
        if nlayer==self.Nlayer-1:
            gamma=np.squeeze(mpo,axis=(3,4))
        else:
            gamma=np.squeeze(mpo).transpose(1,2,0)
        return gamma



    def update(self,nlayer,m,mpo):
        """
        :param nlayer:
        :param m:
        :param mpo:
        :return:
        """

        gamma=self.environment(nlayer,m,mpo)
        l=self.W[nlayer][m].shape
        gamma1=gamma.reshape(np.prod(l[0:-1]),l[-1])
        dmin=min(gamma1.shape)
        u,s,v=np.linalg.svd(gamma1)
        self.W[nlayer][m]=-np.dot(transpose(v),transpose(u)[0:dmin,:]).reshape(l)



    def sweep(self,MPO):
        """

        :param MPO:
        :return:
        """
        s=100
        while s>0:
            for i in range(self.Nlayer):
                for j in range(len(MPO)//(2**(i+1))):
                    self.update(i,j,MPO)
            s-=1
            erg=self.energy(MPO)
            print(erg)



    def magnetization(self,site):
        """

        :return:
        """
        pass

    def energy(self,mpo):


        gamma=self.environment(0,0,mpo)
        erg=tensordot(self.W[0][0],gamma,axes=([0,1,2],[0,1,2]))
        return erg



if __name__=="__main__":
    mpo=MPOIsing.Ising(16,-1,0,0.3)
    tree_tn=tree_TN(4,2,1)
    tree_tn.initialize(len(mpo))
    tree_tn.isometrize()
    tree_tn.sweep(mpo)
    tree_tn.W






