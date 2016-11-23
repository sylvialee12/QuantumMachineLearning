__author__ = 'xlibb'
import numpy as np
import mnist
from numpy import random
from numpy import tensordot
from numpy import transpose

class tnn_classifier():
    """
    """
    def __init__(self,Dbond,d,Dout):
        self.Nlayer= 0
        self.Dbond=Dbond
        self.d=d
        self.Dout=Dout

    def initialize(self,Nx,Ny):
        """

        :param Nx:
        :param Ny:
        :return:
        """

        self.Nlayer=np.log2(Nx).astype('int')
        W=[[] for i in range(self.Nlayer)]
        W[0]=[[random.rand(self.d,self.d,self.d,self.d,self.Dbond) for i in range(Ny//2)] for j in range(Nx//2) ]
        for i in range(1,self.Nlayer-1):
            W[i]=[[random.rand(self.Dbond,self.Dbond,self.Dbond,self.Dbond,self.Dbond) for i in range(Ny//(2**(i+1)))] for j in range(Nx//(2**(i+1)))]
        W[self.Nlayer-1]=[[random.rand(self.Dbond,self.Dbond,self.Dbond,self.Dbond,self.Dout)]]
        self.W=W


    def isometrize(self):
        """
        :return:
        """
        for i in range(self.Nlayer):
            for j in range(len(self.W[i])):
                for k in range(len(self.W[i][j])):
                    l=self.W[i][j][k].shape
                    temp=np.reshape(self.W[i][j][k],[np.prod(l[:-1]),l[-1]])
                    Q,R=np.linalg.qr(temp)
                    self.W[i][j][k]=np.reshape(Q,l)


    def testIsometry(self):
        """
        :return:
        """
        pass


    @staticmethod
    def contract_full(leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor,up_tensor):

        leftup_tem=leftup_tensor.reshape(leftup_tensor.shape+(1,1))
        leftdn_tem=leftdn_tensor.reshape(leftdn_tensor.shape+(1,1))
        rightup_tem=rightup_tensor.reshape(rightup_tensor.shape+(1,1))
        rightdn_tem=rightdn_tensor.reshape(rightdn_tensor.shape+(1,1))

        tem1=tensordot(leftup_tem,leftdn_tem,axes=(-1,-2))
        tem2=tensordot(rightup_tem, rightdn_tem,axes=(-1,-2))
        tem=tensordot(tem1,tem2,axes=([-1,-2],[-1,-2]))
        if up_tensor is None:
            output=tem
        elif len(leftup_tensor.shape)==len(leftdn_tensor.shape)==len(rightup_tensor.shape)==len(rightdn_tensor):
            output=tensordot(up_tensor,tem,axes=([1,2,3,4],[0,1,2,3]))
        else:
            lst=np.array([len(x.shape) for x in [leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor]])
            idx=np.argmax(lst)
            maxlegs=max(lst)
            upleg=list(range(1,5))
            dnleg=list(range(3+maxlegs))
            to_remove=list(range(idx,idx+4))
            upleg_to_con=upleg.remove(idx+1)
            dnleg_to_con=[x for x in dnleg not in to_remove]
            output=tensordot(up_tensor,tem,axes=(upleg_to_con,dnleg_to_con))
        return output


    def environment(self,nlayer,mx,my,data):

        mpo=data
        for i in range(self.Nlayer):
            mpo_i=[]
            for j in range(len(self.W[i])):
                mpo_ij=[]
                for k in range(len(self.W[i][j])):
                    if i!=nlayer or j!=mx or k!=my:
                        output=self.contract_full(mpo[2*j][2*k],mpo[2*j+1][2*k],mpo[2*j][2*k+1],mpo[2*j][2*k+1],self.W[i][j][k])
                        mpo_ij.append(output)
                    else:
                        output=self.contract_full(mpo[2*j][2*k],mpo[2*j+1][2*k],mpo[2*j][2*k+1],mpo[2*j][2*k+1],None)
                        mpo_ij.append(output)
                mpo_i.append(mpo_ij)
            mpo=mpo_i
        return mpo




    def update(self,nlayer,mx,my,data,lvector):

        f=0
        for data_n,l_n in zip(data,lvector):
            gamma_n=self.environment(nlayer,mx,my,data_n)
            ln2=tensordot(self.W[nlayer][mx][my],gamma_n,axes=([0,1,2,3,4],[1,2,3,4,5]))
            f+=tensordot(ln2-l_n,gamma_n,axes=(0,0))
        self.W[nlayer][mx][my]+=f
        shape=self.W[nlayer][mx][my].shape
        tem=self.W[nlayer][mx][my].reshape(np.prod(shape[:-1]),shape[-1])
        Q,R=np.linalg.qr(tem)
        self.W[nlayer][mx][my]=Q.reshape(shape)


    def sweep(self,data,lvector):

        s=50
        while s>0:
            for i in range(self.Nlayer):
                for j in range(len(self.W[i])):
                    for k in range(len(self.W[i][j])):
                        self.update(i,j,k,data,lvector)
            s-=1

    







