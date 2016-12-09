__author__ = 'xlibb'
import numpy as np
import mnist
from numpy import random
from numpy import tensordot
from numpy import transpose
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

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


    def updateWithSVD(self,nlayer,mx,my,data,lvector):

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


    def updateWithLinearE(self,nlayer,mx,my,data,lvector):
        H=0
        B=0
        for data_n,l_n in zip(data,lvector):
            gamma_n=self.environment(nlayer,mx,my,data_n)
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
                gamma_tem=np.reshape(gamma_n,[1,np.prod(gammashape[0:])])
                H_n=np.dot(np.transpose(gamma_tem),gamma_tem)
                B_n=np.dot(np.reshape(l_n,l_n.shape+(1,)),gamma_tem)
                H+=H_n
                B+=B_n
        try:
            solution=B.dot(np.linalg.inv(H))
        except:
            solution=B.dot(np.linalg.inv(H+np.eye(H.shape[0])*10**(-7)))
        if nlayer<self.Nlayer-1:
            Wshape=self.W[nlayer][mx][my].shape
            self.W[nlayer][mx][my]=np.reshape(solution,Wshape)
        else:
            Wshape=self.W[nlayer][mx][my].shape
            solution_tem=np.reshape(solution,(Wshape[-1],)+Wshape[0:-1])
            solution_tem=np.transpose(solution_tem,[1,2,3,4,0])
            self.W[nlayer][mx][my]=np.reshape(solution_tem,Wshape)







    def update(self,nlayer,mx,my,data,lvector):

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
        fshape=f.shape
        ftem=f.reshape(np.prod(fshape[0:-1]),fshape[-1])
        Q,R=np.linalg.qr(ftem)
        deltaW=np.reshape(Q,fshape)
        self.W[nlayer][mx][my]-=deltaW


    def sweep(self,data,lvector,testdata,testlvector):
        s=3
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
                        self.updateWithLinearE(i,j,k,data,lvector)
                        trainlost_i,trainprecision_i=self.test(data,lvector)
                        # testlost_i,testprecision_i=self.test(testdata,testlvector)
                        trainlost.append(trainlost_i)
                        # testlost.append(testlost_i)
                        trainprecision.append(trainprecision_i)
                        # testprecision.append(testprecision_i)

            """
            for i in range(self.Nlayer-1,-1,-1):
                print(i)
                for j in range(len(self.W[i])-1,-1,-1):
                    for k in range(len(self.W[i][j])-1,-1,-1):
                        self.updateWithSVD(i,j,k,data,lvector)
                        # self.update(i,j,k,data,lvector)
                        # self.updateWithLinearE(i,j,k,data,lvector)
                        trainlost_i,trainprecision_i=self.test(data,lvector)
                        testlost_i,testprecision_i=self.test(testdata,testlvector)
                        trainlost.append(trainlost_i)
                        testlost.append(testlost_i)
                        trainprecision.append(trainprecision_i)
                        testprecision.append(testprecision_i)
            """
            s-=1
        return trainlost,testlost,trainprecision,testprecision



    def test(self,testdata,testlvector):
        test_result=[np.argmax(l) for l in testlvector]
        lost,result=self.lostfunc(testdata,testlvector)
        pricision=np.sum(np.array(test_result)==np.array(result))/len(testlvector)
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


if __name__=="__main__":
    Mnist=mnist.load_data()
    margin,pool=2,8
    data,target=mnist.data_process2D(Mnist,margin,pool)
    Dbond,d,Dout=8,2,10
    tnn=tnn_classifier(Dbond,d,Dout)
    train_data,train_target=data[0:2000],target[0:2000]
    test_data,test_target=data[5000:6000],target[5000:6000]
    tnn.initialize(train_data.shape[1],train_data.shape[2])
    tnn.isometrize()
    trainlost,testlost,trainprecision,testprecision=tnn.sweep(train_data,train_target,test_data,test_target)
    plt.figure("Precision")
    plt.plot(trainprecision)
    plt.plot(testprecision)
    plt.savefig("TTN/PrecisionPool"+str(pool)+"Dbond"+str(Dbond)+".png")
    plt.figure("Lost")
    plt.plot(trainlost)
    plt.plot(testlost)
    plt.savefig("TTN/LostPool"+str(pool)+"Dbond"+str(Dbond)+".png")










