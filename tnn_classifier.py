__author__ = 'xlibb'
import __future__
import numpy as np
import mnist
from numpy import random
from numpy import tensordot
from numpy import transpose
from scipy.sparse.linalg import spsolve
import scipy.sparse as ssp
import matplotlib.pyplot as plt
from time import time
import pickle
import warnings




def functimer(func):
    def wrapper(*args,**kwargs):
        now=time()
        num=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time()-now))
        return num
    return wrapper

class tnn_classifier():
    """
    """
    def __init__(self,Dbond,d,Dout):
        self.Nlayer= 0
        self.Dbond=Dbond
        self.d=d
        self.Dout=Dout

    def initialize2(self,Nx,Ny):
        """

        :param Nx:
        :param Ny:
        :return:
        """

        self.Nlayer=np.log2(Nx).astype('int')
        W=[[] for i in range(self.Nlayer)]
        W[0]=[[random.rand(self.d,self.d,self.d,self.d,self.Dbond) for i in range(Ny//2)] for j in range(Nx//2) ]
        for i in range(1,self.Nlayer-1):
            W[i]=[[random.rand(i*self.Dbond,i*self.Dbond,i*self.Dbond,i*self.Dbond,(i+1)*self.Dbond) for l in range(Ny//(2**(i+1)))] for j in range(Nx//(2**(i+1)))]
        W[self.Nlayer-1]=[[random.rand((self.Nlayer-1)*self.Dbond,(self.Nlayer-1)*self.Dbond,(self.Nlayer-1)*self.Dbond,(self.Nlayer-1)*self.Dbond,self.Dout)]]
        self.W=W

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
            W[i]=[[random.rand(self.Dbond,self.Dbond,self.Dbond,self.Dbond,self.Dbond) for l in range(Ny//(2**(i+1)))] for j in range(Nx//(2**(i+1)))]
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
        isometry=[]
        for j in range(len(self.W[0])):
            isometry_ij=[]
            for k in range(len(self.W[0][j])):
                tem=tensordot(self.W[0][j][k],np.conj(self.W[0][j][k]),axes=([0,1,2,3],[0,1,2,3]))
                isometry_ij.append(tem)
            isometry.append(isometry_ij)

        for i in range(1,self.Nlayer):
            isometry_i=[]
            for j in range(len(self.W[i])):
                isometry_ij=[]
                for k in range(len(self.W[i][j])):
                    output=self.contract_conj(isometry[2*j][2*k],isometry[2*j+1][2*k],isometry[2*j][2*k+1],isometry[2*j+1][2*k+1],self.W[i][j][k],self.W[i][j][k])
                    isometry_ij.append(output)
                isometry_i.append(isometry_ij)
            isometry=isometry_i
        contracted=isometry[0][0]
        return contracted



    @staticmethod
    def contract_full(leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor,up_tensor):

        lst=[len(x.shape) for x in [leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor]]


        leftup_tem=leftup_tensor.reshape(leftup_tensor.shape+(1,1))
        leftdn_tem=leftdn_tensor.reshape(leftdn_tensor.shape+(1,1))
        rightup_tem=rightup_tensor.reshape(rightup_tensor.shape+(1,1))
        rightdn_tem=rightdn_tensor.reshape(rightdn_tensor.shape+(1,1))

        tem1=tensordot(leftup_tem,leftdn_tem,axes=(-1,-2))
        tem2=tensordot(rightup_tem, rightdn_tem,axes=(-1,-2))
        if up_tensor is None:
            tem=tensordot(tem1,tem2,axes=([-1,-3],[-1,-3]))
            output=tem
        elif lst.count(lst[0])==len(lst):
            tem=tensordot(tem1,tem2,axes=([-1,-3],[-1,-3]))
            output=tensordot(up_tensor,tem,axes=([0,1,2,3],[0,1,2,3]))
        else:
            lst=np.array(lst)
            idx=np.argmax(lst)
            maxlegs=max(lst)
            if idx<2:
                tem=tensordot(tem1,tem2,axes=([-1,-3-(maxlegs-1)*idx],[-1,-3]))
            else:
                tem=tensordot(tem1,tem2,axes=([-1,-3],[-1,-3-(maxlegs-1)*(idx-2)]))
            if maxlegs==4:
                upleg=list(range(0,4))
                dnleg=list(range(3+maxlegs))
                to_remove=list(range(idx,idx+maxlegs))
                upleg_to_con=[x for x in upleg if x!=idx]
                dnleg_to_con=[x for x in dnleg if x not in to_remove]
                tem=tensordot(up_tensor,tem,axes=(upleg_to_con,dnleg_to_con))
                output=tem.transpose(1,0,2,3,4,5)
            elif maxlegs==6:
                upleg=list(range(0,4))
                dnleg=list(range(3+maxlegs))
                to_remove=list(range(idx+1,idx+6))
                upleg_to_con=upleg
                dnleg_to_con=[x for x in dnleg if x not in to_remove]
                output=tensordot(up_tensor,tem,axes=(upleg_to_con,dnleg_to_con))
            else:
                print("Error")
        return output

    @staticmethod
    def contract_full2(leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor,up_tensor):
        """

        :param leftup_tensor:
        :param leftdn_tensor:
        :param rightup_tensor:
        :param rightdn_tensor:
        :param up_tensor:
        :return:
        """

        lst1=[leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor]
        if "cone" in lst1:
            idx=lst1.index("cone")
            to_contract=[x for x in range(4) if x!=idx]
            lst2=[x.reshape(x.shape+(1,)) for x in lst1 if x!="cone"]
            tem=up_tensor
            for i in range(3):
                tem1=tensordot(tem,lst2[i],axes=(to_contract[i],0))
                axeorder=list(range(to_contract[i]))+[4]+list(range(to_contract[i],4))
                tem=transpose(tem1,axeorder)
            output=tem.squeeze().transpose()
        else:
            numlst=[len(x.shape) for x in lst1]
            if numlst.count(numlst[0])==len(numlst):
                tem=up_tensor
                for i in range(4):
                    tem1=tensordot(tem,lst1[i].reshape(lst1[i].shape+(1,)),axes=(i,0))
                    axeorder=list(range(i))+[4]+list(range(i,4))
                    tem=transpose(tem1,axeorder)
                output=tem.squeeze()
            else:
                idx=np.argmax(np.array(numlst))
                contractfirst=[x for x in range(4) if x!=idx]
                tem=up_tensor
                for i in range(3):
                    inx=contractfirst[i]
                    tem1=tensordot(tem,lst1[inx].reshape(lst1[inx].shape+(1,)),axes=(inx,0))
                    axeorder=list(range(inx))+[4]+list(range(inx,4))
                    tem=transpose(tem1,axeorder)
                output=tensordot(tem,lst1[idx],axes=(idx,0))
        return output.squeeze()

    @staticmethod
    def contract_conj(leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor,up_tensor,dn_tensor):

        leftup_tem=leftup_tensor.reshape(leftup_tensor.shape+(1,1))
        leftdn_tem=leftdn_tensor.reshape(leftdn_tensor.shape+(1,1))
        rightup_tem=rightup_tensor.reshape(rightup_tensor.shape+(1,1))
        rightdn_tem=rightdn_tensor.reshape(rightdn_tensor.shape+(1,1))

        tem1=tensordot(leftup_tem,leftdn_tem,axes=(-1,-2))
        tem2=tensordot(rightup_tem, rightdn_tem,axes=(-1,-2))
        tem=tensordot(tem1,tem2,axes=([-1,-4],[-1,-4]))
        tem=tensordot(up_tensor,tem,axes=([0,1,2,3],[0,2,4,6]))
        output=tensordot(tem,np.conj(dn_tensor),axes=([1,2,3,4],[0,1,2,3]))
        return output

    @staticmethod
    def contract_conj3(leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor,up_tensor):
        """

        :param leftup_tensor:
        :param leftdn_tensor:
        :param rightup_tensor:
        :param rightdn_tensor:
        :param up_tensor:
        :return:
        """
        lst1=[leftup_tensor,leftdn_tensor,rightup_tensor,rightdn_tensor]
        if "cone" in lst1:
            idx=lst1.index("cone")
            to_contract=[x for x in range(4) if x!=idx]
            lst2=[x for x in lst1 if x!="cone"]
            tem=up_tensor
            for i in range(3):
                tem1=tensordot(tem,lst2[i],axes=(to_contract[i],0))
                axeorder=list(range(to_contract[i]))+[4]+list(range(to_contract[i],4))
                tem=transpose(tem1,axeorder)
            output_tem=tensordot(tem,np.conj(up_tensor),axes=(to_contract,to_contract))
            output=transpose(output_tem,[1,3,0,2])
        else:
            numlst=[len(x.shape) for x in lst1]
            if numlst.count(numlst[0])==len(numlst):
                tem=up_tensor
                for i in range(4):
                    tem1=tensordot(tem,lst1[i],axes=(i,0))
                    axeorder=list(range(i))+[4]+list(range(i,4))
                    tem=transpose(tem1,axeorder)
                output_tem=tensordot(tem,np.conj(up_tensor),axes=([0,1,2,3],[0,1,2,3]))
                output=output_tem
            else:
                idx=np.argmax(np.array(numlst))
                contractfirst=[x for x in range(4) if x!=idx]
                tem=up_tensor
                for i in range(3):
                    tem1=tensordot(tem,lst1[contractfirst[i]],axes=(contractfirst[i],0))
                    axeorder=list(range(contractfirst[i]))+[4]+list(range(contractfirst[i],4))
                    tem=transpose(tem1,axeorder)
                tem=tensordot(tem,np.conj(up_tensor),axes=(contractfirst,contractfirst))
                tem=transpose(tem,[1,3,0,2])
                output=tensordot(tem,lst1[idx],axes=([2,3],[0,1]))
        return output


    def environment(self,nlayer,mx,my,data):
        """

        :param nlayer:
        :param mx:
        :param my:
        :param data:
        :return:
        """
        mpo=data
        for i in range(self.Nlayer):
            mpo_i=[]
            for j in range(len(self.W[i])):
                mpo_ij=[]
                for k in range(len(self.W[i][j])):
                    if i!=nlayer or j!=mx or k!=my:
                        output=self.contract_full(mpo[2*j][2*k],mpo[2*j+1][2*k],mpo[2*j][2*k+1],mpo[2*j+1][2*k+1],self.W[i][j][k])
                        mpo_ij.append(output)
                    else:
                        output=self.contract_full(mpo[2*j][2*k],mpo[2*j+1][2*k],mpo[2*j][2*k+1],mpo[2*j+1][2*k+1],None)
                        mpo_ij.append(output)
                mpo_i.append(mpo_ij)
            mpo=mpo_i
        return mpo[0][0]





    def environment2(self,nlayer,mx,my,data):
        """

        :param nlayer:
        :param mx:
        :param my:
        :param data:
        :return:
        """
        mpo=data
        for i in range(self.Nlayer):
            mpo_i=[]
            for j in range(len(self.W[i])):
                mpo_ij=[]
                for k in range(len(self.W[i][j])):
                    if i!=nlayer or j!=mx or k!=my:
                        output=self.contract_full2(mpo[2*j][2*k],mpo[2*j+1][2*k],mpo[2*j][2*k+1],mpo[2*j+1][2*k+1],self.W[i][j][k])
                        mpo_ij.append(output)
                    else:
                        coneinside=[mpo[2*j][2*k],mpo[2*j+1][2*k],mpo[2*j][2*k+1],mpo[2*j+1][2*k+1]]
                        mpo_ij.append("cone")
                mpo_i.append(mpo_ij)
            mpo=mpo_i

        coneoutside=mpo[0][0]
        if nlayer!=self.Nlayer-1:
            tem=coneoutside
            for cone_i in coneinside:
                cone_tem=cone_i.reshape(cone_i.shape+(1,))
                tem=tem.reshape(tem.shape+(1,))
                tem=tensordot(tem,cone_tem,axes=(-1,-1))
            output=tem.squeeze()
        else:
            tem=np.array(1)
            for cone_i in coneinside:
                cone_tem=cone_i.reshape(cone_i.shape+(1,))
                tem=tem.reshape(tem.shape+(1,))
                tem=tensordot(tem,cone_tem,axes=(-1,-1))
            output=tem.squeeze()
        return output



    def environmentL2(self,nlayer,mx,my):
        """

        :param nlayer:
        :param mx:
        :param my:
        :return:
        """
        mpo=[[np.eye(2) for i in range(2**self.Nlayer)] for j in range(2**self.Nlayer)]
        for i in range(self.Nlayer):
            mpo_i=[]
            for j in range(len(self.W[i])):
                mpo_ij=[]
                for k in range(len(self.W[i][j])):
                    if i!=nlayer or j!=mx or k!=my:
                        output=self.contract_conj3(mpo[2*j][2*k],mpo[2*j+1][2*k],mpo[2*j][2*k+1],mpo[2*j+1][2*k+1],self.W[i][j][k])
                        mpo_ij.append(output)
                    else:
                        coneinside=[mpo[2*j][2*k],mpo[2*j+1][2*k],mpo[2*j][2*k+1],mpo[2*j+1][2*k+1]]
                        mpo_ij.append("cone")
                mpo_i.append(mpo_ij)
            mpo=mpo_i
        if nlayer!=self.Nlayer-1:
            coneoutside=np.trace(mpo[0][0])
            tem=coneoutside
            for cone_i in coneinside:
                cone_tem=cone_i.reshape(cone_i.shape+(1,))
                tem=tem.reshape(tem.shape+(1,))
                tem=tensordot(tem,cone_tem,axes=(-1,-1))
            output=transpose(tem,[0,2,4,6,8,1,3,5,7,9])
        else:
            tem=np.array(1)
            for cone_i in coneinside:
                cone_tem=cone_i.reshape(cone_i.shape+(1,))
                tem=tem.reshape(tem.shape+(1,))
                tem=tensordot(tem,cone_tem,axes=(-1,-1))
            output=transpose(tem,[0,2,4,6,1,3,5,7])
        return output


    @functimer
    def updateWithSVD(self,nlayer,mx,my,data,lvector):

        f=0
        for data_n,l_n in zip(data,lvector):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gamma_n=self.environment2(nlayer,mx,my,data_n)
            if nlayer<self.Nlayer-1:
                ln2=tensordot(self.W[nlayer][mx][my],gamma_n,axes=([4,0,1,2,3],[1,2,3,4,5]))
                f+=tensordot(ln2-l_n,gamma_n,axes=(0,0))
            else:
                ln2=tensordot(self.W[nlayer][mx][my],gamma_n,axes=([0,1,2,3],[0,1,2,3]))
                f+=np.kron(gamma_n,ln2-l_n).reshape(self.W[nlayer][mx][my].shape)
        if nlayer<self.Nlayer-1:
            f=f.transpose(1,2,3,4,0)
        fshape=f.shape
        gamma1=np.reshape(f,[np.prod(fshape[0:-1]),fshape[-1]])
        dmin=min(gamma1.shape)
        try:
            u,s,v=np.linalg.svd(gamma1)
        except:
            gamma1
        tem=-np.dot(np.conj(transpose(v)),np.conj(transpose(u))[0:dmin,:])
        tem=transpose(tem)
        self.W[nlayer][mx][my]=tem.reshape(fshape)



    @functimer
    def updateWithLinearE2(self,nlayer,mx,my,data,lvector,lamb):
        """
        :param nlayer:
        :param mx:
        :param my:
        :param data:
        :param lvector:
        :return:
        """
        H=0
        B=0
        t0=time()
        for data_n,l_n in zip(data,lvector):
            gamma_n=self.environment2(nlayer,mx,my,data_n)
            if nlayer<self.Nlayer-1:
                gammashape=gamma_n.shape
                gamma_n=np.transpose(gamma_n,[0,2,3,4,5,1])
                gamma_tem=np.reshape(gamma_n,[gammashape[0],np.prod(gammashape[1:])])
                H_n=np.dot(np.transpose(gamma_tem),gamma_tem)
                B_n=np.dot(l_n,gamma_tem)
                H+=H_n
                B+=B_n
            else:
                gammashape=gamma_n.shape
                gamma_tem=np.reshape(gamma_n,[np.prod(gammashape[0:]),1])
                H_n=np.dot(gamma_tem,np.transpose(gamma_tem))
                B_n=np.dot(gamma_tem,np.reshape(l_n,(1,)+l_n.shape))
                H+=H_n
                B+=B_n
        print(time()-t0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gamma1=self.environmentL2(nlayer,mx,my)
        lenshape=len(gamma1.shape)
        H1=lamb*np.reshape(gamma1,[np.prod(gamma1.shape[0:lenshape//2]),np.prod(gamma1.shape[lenshape//2:])])
        try:
            H=ssp.csc_matrix(H+H1)
            B=ssp.csc_matrix(B)
            solution=spsolve(H,B)
        except:
            H=ssp.csc_matrix(H+H1+np.eye(H.shape[0])*1e-8)
            B=ssp.csc_matrix(B)
            solution=spsolve(H,B)
        if nlayer<self.Nlayer-1:
            Wshape=self.W[nlayer][mx][my].shape
            self.W[nlayer][mx][my]=np.reshape(solution,Wshape)
        else:
            Wshape=self.W[nlayer][mx][my].shape
            self.W[nlayer][mx][my]=np.reshape(solution.toarray(),Wshape)


    @functimer
    def updateWithLinearE(self,nlayer,mx,my,data,lvector):
        """
        :param nlayer:
        :param mx:
        :param my:
        :param data:
        :param lvector:
        :return:
        """
        t0=time()
        environment=self.environment

        gamma=np.array(list(map(lambda datan:environment(nlayer,mx,my,datan),data)))
        print(time()-t0)
        if nlayer<self.Nlayer-1:
            gamma=np.transpose(gamma,[3,4,5,6,2,1,0])
            gamma_tem=np.reshape(gamma,[np.prod(gamma.shape[0:-2]),gamma.shape[-2],gamma.shape[-1]])
            H=tensordot(gamma_tem,gamma_tem,axes=([1,2],[1,2]))
            B=tensordot(gamma_tem,lvector,axes=([1,2],[1,0]))
            B=B.reshape(B.shape[0],1)
        else:
            gamma_tem=np.reshape(gamma,[np.prod(gamma.shape[1:]),gamma.shape[0]])
            H=gamma_tem.dot(transpose(gamma_tem))
            B=gamma_tem.dot(lvector)


        H=ssp.csc_matrix(H+np.eye(H.shape[0])*1e-8)
        B=ssp.csc_matrix(B)
        solution=spsolve(H,B)

        if nlayer<self.Nlayer-1:
            Wshape=self.W[nlayer][mx][my].shape
            self.W[nlayer][mx][my]=np.reshape(solution,Wshape)
        else:
            Wshape=self.W[nlayer][mx][my].shape
            solution_tem=np.reshape(solution.toarray(),Wshape)
            self.W[nlayer][mx][my]=solution_tem


    @functimer
    def updateWithLinearEWithL2(self,nlayer,mx,my,data,lvector,lamb):
        """
        :param nlayer:
        :param mx:
        :param my:
        :param data:
        :param lvector:
        :return:
        """
        t0=time()
        environment=self.environment2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gamma=np.array(list(map(lambda datan: environment(nlayer, mx, my, datan), data)))
        print(time()-t0)
        print(gamma.max()-gamma.min())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gamma1=self.environmentL2(nlayer,mx,my)
        if nlayer<self.Nlayer-1:
            gamma=np.transpose(gamma,[3,4,5,6,2,1,0])
            gamma_tem=np.reshape(gamma,[np.prod(gamma.shape[0:-2]),gamma.shape[-2],gamma.shape[-1]])
            H=tensordot(gamma_tem,gamma_tem,axes=([1,2],[1,2]))
            B=tensordot(gamma_tem,lvector,axes=([1,2],[1,0]))
            B=B.reshape(B.shape[0],1)
            H1=lamb*gamma1.reshape([np.prod(gamma1.shape[0:5]),np.prod(gamma1.shape[5:10])])

        else:
            gamma_tem=np.reshape(gamma,[np.prod(gamma.shape[1:]),gamma.shape[0]])
            H=gamma_tem.dot(transpose(gamma_tem))
            B=gamma_tem.dot(lvector)
            H1=lamb*gamma1.reshape([np.prod(gamma1.shape[0:4]),np.prod(gamma1.shape[4:8])])

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                H=ssp.csc_matrix(H+H1)
                B=ssp.csc_matrix(B)
                solution=spsolve(H,B)
            except:
                H=ssp.csc_matrix(H+H1+1e-08*np.eye(H.shape[0]))
                B=ssp.csc_matrix(B)
                solution=spsolve(H,B)

        if nlayer<self.Nlayer-1:
            Wshape=self.W[nlayer][mx][my].shape
            solution_tem=np.reshape(solution,[np.prod(Wshape[0:-1]),Wshape[-1]])
            self.W[nlayer][mx][my]=np.reshape(solution_tem,Wshape)

        else:
            Wshape=self.W[nlayer][mx][my].shape
            self.W[nlayer][mx][my]=np.reshape(solution.toarray(),Wshape)



    @functimer
    def update(self,nlayer,mx,my,data,lvector):
        """
        :param nlayer:
        :param mx:
        :param my:
        :param data:
        :param lvector:
        :return:
        """

        f=0
        for data_n,l_n in zip(data,lvector):
            gamma_n=self.environment(nlayer,mx,my,data_n)
            if nlayer<self.Nlayer-1:
                ln2=tensordot(self.W[nlayer][mx][my],gamma_n,axes=([4,0,1,2,3],[1,2,3,4,5]))
                f+=tensordot(ln2-l_n,gamma_n,axes=(0,0))
            else:
                ln2=tensordot(self.W[nlayer][mx][my],gamma_n,axes=([0,1,2,3],[0,1,2,3]))
                f+=np.kron(gamma_n,ln2-l_n).reshape(self.W[nlayer][mx][my].shape)
        if nlayer<self.Nlayer-1:
            f=f.transpose(1,2,3,4,0)
        deltaW=f
        self.W[nlayer][mx][my]-=deltaW


    def sweep(self,data,lvector,testdata,testlvector,lamb):
        s=5
        trainlost=[]
        testlost=[]
        trainprecision=[]
        testprecision=[]
        while s>0:
            for i in range(self.Nlayer):
                print(i)
                for j in range(len(self.W[i])):
                    for k in range(len(self.W[i][j])):

                        # self.updateWithSVD(i,j,k,data,lvector)
                        # self.update(i,j,k,data,lvector)
                        if i<self.Nlayer-1:
                            self.updateWithLinearEWithL2(i,j,k,data,lvector,lamb)
                        else:
                            self.updateWithLinearE2(i,j,k,data,lvector,lamb)
                        trainlost_i,trainprecision_i=self.test(data,lvector,lamb)
                        testlost_i,testprecision_i=self.test(testdata,testlvector,lamb)
                        trainlost.append(trainlost_i)
                        testlost.append(testlost_i)
                        trainprecision.append(trainprecision_i)
                        testprecision.append(testprecision_i)

            s-=1
        with open("trainlost.txt","wb") as f:
            pickle.dump(trainlost,f)
        with open("trainpreci.txt","wb") as f2:
            pickle.dump(trainprecision,f2)
        with open("testlost.txt","wb") as f3:
            pickle.dump(testlost,f3)
        with open("testpreci.txt","wb") as f4:
            pickle.dump(testprecision,f4)
        return trainlost,testlost,trainprecision,testprecision


    @functimer
    def test(self,testdata,testlvector,lamb):
        test_result=testlvector.argmax(axis=1)
        lost,result=self.lossWithL2(testdata,testlvector,lamb)
        pricision=np.sum(test_result==result)/len(testlvector)
        return lost,pricision



    def lostfunc(self,data,lvector):
        lost=0
        result=[]
        Nnum=len(lvector)
        for data_n,l_n in zip(data,lvector):
            gamma=self.environment(self.Nlayer-1,0,0,data_n)
            ln2=tensordot(self.W[self.Nlayer-1][0][0],gamma,axes=([0,1,2,3],[0,1,2,3]))
            lost+=sum((ln2-l_n)**2)/Nnum
            result.append(np.argmax(ln2))

        return lost,result


    def lostfunc2(self,data,lvector):
        Nnum=len(lvector)
        environment=self.environment
        gamma=np.array([environment(self.Nlayer-1,0,0,datan) for datan in data])
        ln2=tensordot(gamma,self.W[self.Nlayer-1][0][0],axes=([1,2,3,4],[0,1,2,3]))
        lost=np.sum((ln2-lvector)**2)/Nnum
        result=ln2.argmax(axis=1)
        return lost,result


    def lossWithL2(self,data,lvector,lamb):
        Nnum=len(lvector)
        environment=self.environment2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gamma=np.array([environment(self.Nlayer-1,0,0,datan) for datan in data])
        ln2=tensordot(gamma,self.W[self.Nlayer-1][0][0],axes=([1,2,3,4],[0,1,2,3]))
        lost=np.sum((ln2-lvector)**2)/Nnum+lamb*np.sum(self.testIsometry())
        result=ln2.argmax(axis=1)
        return lost,result


    def savenetwork(self,filename):
        """
        :return:
        """
        with open(filename,"wb") as f:
            pickle.dump(self.W,f)


    def loadnetwork(self,filename):
        with open(filename,"rb") as f:
            self.W=pickle.load(f)


if __name__=="__main__":
    Mnist=mnist.load_data()
    margin,pool=2,4
    data,target=mnist.data_process2D(Mnist,margin,pool,"cosine")
    Dbond,d,Dout=6,2,10
    lamb=0
    tnn=tnn_classifier(Dbond,d,Dout)
    trainend,testend=40000,50000
    train_data,train_target=data[0:trainend],target[0:trainend]
    test_data,test_target=data[trainend:testend],target[trainend:testend]
    tnn.initialize(train_data.shape[1],train_data.shape[2])
    tnn.isometrize()
    trainlost,testlost,trainprecision,testprecision=tnn.sweep(train_data,train_target,test_data,test_target,lamb)
    tnn.savenetwork("WtrainedWithLinearE.txt")
    plt.figure("Precision")
    plt.plot(trainprecision)
    plt.plot(testprecision)
    plt.savefig("TTN/4PrecisionPool"+str(pool)+"Dbond"+str(Dbond)+".png")
    plt.figure("Lost")
    plt.plot(trainlost)
    plt.plot(testlost)
    plt.savefig("TTN/4LostPool"+str(pool)+"Dbond"+str(Dbond)+".png")


