__author__ = 'xlibb'
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import copy
import mnist
from time import time
import csv


def functimer(func):
    def wrapper(*args,**kwargs):
        now=time()
        num=func(*args,**kwargs)
        print("Running %s : %f s"%(func.__name__,time()-now))
        return num
    return wrapper


class tn_classifier():
    """
    This is a classifer with the algorithm of tensor network
    """

    def __init__(self,N,D,n,eps):
        """
        :param data: the imported datasets
        :param l: the site with labeling

        """
        self.D=D  # bond dimension of the tensor network
        self.n=n  # the number of classification
        self.N = N #
        self.d=2
        self.eps=eps


    def initialize(self):
        """
        Initialize tensors A, choosing the canonical gauge(QR decomposition)
        :return: randomized weight tensor A
        """
        N=self.N
        A=[[] for i in range(N)]
        A[0]=random.rand(self.d,1,self.D,self.n)
        tem=np.transpose(A[0],[0,1,3,2])
        A[0]=np.reshape(tem,[self.d*1*self.n,self.D])
        dmin=min(A[0].shape)
        Q,R=np.linalg.qr(A[0])
        A[0]=np.reshape(Q,[self.d,1,self.n,dmin])
        A[0]=np.transpose(A[0],[0,1,3,2])
        for i in range(1,N-1):
            A[i]=random.rand(self.d,self.D,self.D)
            dr1,dr2=np.shape(R)
            tem=np.tensordot(R,A[i],axes=(1,1))
            tem=np.transpose(tem,[1,0,2])
            tem=np.reshape(tem,[self.d*dr1,self.D])
            Q,R=np.linalg.qr(tem)
            dmin=min(tem.shape)
            A[i]=np.reshape(Q,[self.d,dr1,dmin])

        A[N-1]=random.rand(self.d,self.D,1)
        dr1,dr2=np.shape(R)
        tem=np.tensordot(R,A[N-1],axes=(1,1))
        tem=np.transpose(tem,[1,0,2])
        tem=np.reshape(tem,[self.d*dr1,1])
        Q,R=np.linalg.qr(tem)
        dmin=min(tem.shape)
        A[N-1]=np.reshape(Q,[self.d,dr1,1])
        norm=self.normalization(A)
        A=[a/np.sqrt(norm) for a in A]
        return A


    def normalization(self,A):
        """
        :param A: To get the normalization factor of A
        :return:
        """
        norm0=np.tensordot(A[0],np.conj(A[0]),axes=([0,3],[0,3]))
        norm0=np.transpose(norm0,[0,2,1,3])
        for i in range(1,len(A)):
            normi=np.tensordot(A[i],np.conj(A[i]),axes=(0,0))
            normi=np.transpose(normi,[0,2,1,3])
            norm0=np.tensordot(norm0,normi,axes=([2,3],[0,1]))
        norm0=np.squeeze(norm0)
        norm=norm0**(1/len(A))
        return norm


    def normalization2(self,A):
        """
        :param A: To get the normalization factor of A
        :return:
        """
        norm0=np.array([1]).reshape([1,1,1,1])
        for Ai in A:
            if Ai.ndim<4:
                normi=np.tensordot(Ai,np.conj(Ai),axes=(0,0))
            else:
                normi=np.tensordot(Ai,np.conj(Ai),axes=([0,3],[0,3]))
            normi=np.transpose(normi,[0,2,1,3])
            norm0=np.tensordot(norm0,normi,axes=([2,3],[0,1]))
        norm0=np.squeeze(norm0)
        norm=norm0**(1/len(A))
        return norm


    def normalize(self,A):
        """
        :param A:
        :return:
        """
        norm=self.normalization2(A)
        A=[Ai/np.sqrt(norm) for Ai in A]
        return A



    def right_getB(self,A,l):
        """
        :param A:
        :return:
        """
        B=np.tensordot(A[l],A[l+1],axes=(2,1))
        B=np.transpose(B,[0,3,1,4,2])
        return B

    def left_getB(self, A, l):
        """
        :param A:
        :return:
        """
        B = np.tensordot(A[l], A[l + 1], axes=(2, 1))
        B = np.transpose(B, [0, 2, 1, 3, 4])
        return B


    def getphi(self,A,phi,l):
        """
        :param A:
        :param phi:
        :return:
        """

        phi0 = np.array([1])
        for i in range(l):
            phi1 = np.tensordot(A[i], phi[i], axes=(0, 0))
            phi0 = np.tensordot(phi1, phi0, axes=(0, -1))

        phi2 = np.array([1])
        for i in range(self.N - 1, l+1, -1):
            phi1 = np.tensordot(A[i], phi[i], axes=(0, 0))
            phi2 = np.tensordot(phi1, phi2, axes=(1, 0))

        tilde_phi = [ phi[l], phi[l + 1], phi0,phi2]

        return tilde_phi


    def gradient(self,B,tilde_phi,l_vector,step=1):
        """
        This is to find out the gradient of the lost function versus B
        :param B:
        :param tilde_phi:
        :return:
        """
        d0,d1,d2,d3,d4=B.shape

        j=0
        s=1
        while s==1 and j<step:
            c = 0
            for (phi,l) in zip(tilde_phi,l_vector):
                f = copy.deepcopy(B)
                for i in range(4):
                    f=np.tensordot(f,phi[i],axes=(0,0))
                temp=np.kron(phi[0],phi[1])
                temp=np.kron(temp,phi[2])
                temp=np.kron(temp,phi[3])
                temp=np.kron(temp,l-f)
                temp=np.reshape(temp,[d0,d1,d2,d3,d4])
                c+=temp
            B+=c
            j+=1

        return B


    def right_updateA(self,updatedB,A,l):
        """
        :param updatedB:
        :param A:
        :param l:
        :return:
        """

        d0,d1,d2,d3,d4=updatedB.shape
        B=np.transpose(updatedB,[0,2,1,3,4])
        B=np.reshape(B,[d0*d2,d1*d3*d4])
        d=min(min(B.shape),self.D)
        try:
            u,s,v=np.linalg.svd(B)
        except:
            B
        u=u[:,:d]
        s=s[:d]/s[0]
        v=v[:d,:]
        # u,s,v=ssl.svds(B,d)
        A[l]=u.reshape([d0,d2,d])
        temp=np.tensordot(np.diag(s),v,axes=(1,0))
        A[l+1]=temp.reshape([d,d1,d3,d4])
        A[l+1]=np.transpose(A[l+1],[1,0,2,3])
        return A


    def right_updatephi(self,updatedA,phi,tildephio,l):
        temp = np.tensordot(updatedA[l], tildephio[0], axes=(0, 0))
        tildephio[2] = np.tensordot(tildephio[2], temp, axes=(-1, 0))
        tildephio[0] = phi[l+1]
        tildephio[1] = phi[l + 2]
        temp2 = np.tensordot(updatedA[l + 2], phi[l + 2], axes=(0, 0))
        try:
            tildephio[3]=np.tensordot(np.linalg.inv(temp2),tildephio[3],axes=(1,0))
        except:
            tildephio=self.getphi(updatedA,phi,l+1)
        return tildephio

    def left_updatephi(self,updatedA,phi,tildephio,l):
        """
        :param updatedA:
        :param phi:
        :param tildephio:
        :param l:
        :return:
        """
        temp=np.tensordot(updatedA[l+1],tildephio[1],axes=(0,0))
        tildephio[3]=np.tensordot(temp,tildephio[3],axes=(1,0))
        tildephio[1]=phi[l]
        tildephio[0]=phi[l-1]
        temp2 = np.tensordot(updatedA[l -1], phi[l -1], axes=(0, 0))
        try:
            tildephio[2]=np.tensordot(np.linalg.inv(temp2),tildephio[2],axes=(0,0))
        except:
            tildephio=self.getphi(updatedA,phi,l-1)
        return tildephio


    def left_updateA(self,updatedB,A,l):
        """
        The weight tensor A
        :param updatedB:
        :param A:
        :param l:
        :return:
        """
        d0,d1,d2,d3,d4=updatedB.shape
        B=np.transpose(updatedB,[0,2,4,1,3])
        B=np.reshape(B,[d0*d2*d4,d1*d3])
        d = min(min(B.shape), self.D)
        u,s,v=np.linalg.svd(B)
        u=u[:,:d]
        s=s[:d]/(s[0])
        v=v[:d,:]
        A[l+1]=v.reshape([d,d1,d3])
        A[l+1]=np.transpose(A[l+1],[1,0,2])
        temp=np.tensordot(u,np.diag(s),axes=(1,0))
        A[l]=temp.reshape([d0,d2,d4,d])
        A[l]=np.transpose(A[l],[0,1,3,2])
        return A

    def get_phi_set(self,data):
        Nt,N,d0=data.shape
        phi_tilde=[[] for i in range(Nt)]
        getphi=self.getphi
        for i in range(Nt):
            phi_tilde[i]=getphi(A,data[i],0)
        return phi_tilde

    def right_sweep(self,A,phi_tilde,data,target,i):
        """
        Sweep a step from i to i+1
        :param A: Input weight parameter tensors
        :param phi_tilde: input phi_tilde
        :param data: training data
        :param target: training target
        :param i: the position of the output attached leg
        :return:updated weight parameter tensor A and updated phi_tilde after sweep a step towards right hand sight
        """
        print(i)
        B = self.right_getB(A, i)
        B=self.gradient(B,phi_tilde,target)
        maxB=B.max()
        A=self.right_updateA(B,A,i)
        if i<len(A)-2:
            # phi_tilde=[self.right_updatephi(A,data_i,phi_tilde_i,i) for data_i,phi_tilde_i in zip(data,phi_tilde)]
            phi_tilde=[self.getphi(A,data_i,i+1) for data_i in data]
        return A,phi_tilde


    def left_sweep(self,A,phi_tilde,data,target,i):
        """
        Sweep a step from i to i-1
        :param A:
        :param phi_tilde:
        :param data:
        :param target:
        :param i:
        :return:
        """
        print(i)
        B = self.left_getB(A, i)
        B=self.gradient(B,phi_tilde,target)
        A=self.left_updateA(B,A,i)
        if i>0:
            # for j in range(Nt):
            #     phi_tilde[j]=self.left_updatephi(A,data[j],phi_tilde[j],i)
            phi_tilde=[self.getphi(A,data_i,i-1) for data_i in data]
        return A,phi_tilde



    def test(self,A,testdata,testvector):
        result=[]
        for (data,target) in zip(testdata,testvector):
            to = np.array([[1]])
            for i in range(len(A)):
                ti=np.tensordot(A[i],data[i],axes=(0,0))
                to=np.tensordot(to,ti,axes=(1,0))
            result.append(np.argmax(np.squeeze(to)))

        return result


    def test2(self,A,testdata,testvector,l):
        """
        Test the testdata after well-trained A.
        :param A: Weight tensor after training
        :param testdata: data to be test
        :param testvector: data target to be test
        :param l: the position of the attached output leg
        :return:the test result
        """
        result=[]
        B=self.right_getB(A,l)
        for (data,target) in zip(testdata,testvector):
            phi_tilde=self.getphi(A,data,l)
            f=np.tensordot(B,phi_tilde[0],axes=(0,0))
            f=np.tensordot(f,phi_tilde[1],axes=(0,0))
            f=np.tensordot(f,phi_tilde[2],axes=(0,0))
            f=np.tensordot(f,phi_tilde[3],axes=(0,0))
            result.append(np.argmax(np.squeeze(f)))
        return result

    def write_A(self,filename,A):
        """
        Write the result into a .csv file
        :return:
        """
        with open(filename,'wb') as f:
            wr=csv.writer(f)
            for item in A:
                wr.writerow(A)

    def write_result(self,filename,result):
        """
        Write the classification result into filename file.
        :param filename:
        :param result:
        :return:
        """
        pass


    def write_precision(self,filename,precision):
        """
        Write the precision result into filename file
        :param filename:
        :param precision:
        :return:
        """
        precision.tofile(filename,sep=",",format="%10.5f")


if __name__=="__main__":
    Mnist=mnist.load_data()
    data,target=mnist.data_process(Mnist,4)
    a=tn_classifier(len(data[0]),10,10,10**(-6))
    A=a.initialize()
    phi_tilde=a.get_phi_set(data[0:2000])
    test_target=[np.argmax(l) for l in target[6000:7000]]
    precision=np.zeros(len(A)-1)
    for i in range(len(A)-1):
        if i>100:
            a
        result2=a.test2(A,data[6000:7000],target[6000:7000],i)
        precision[i]=sum(np.array(result2)==np.array(test_target))*1.0/len(test_target)
        A,phi_tilde=a.right_sweep(A,phi_tilde,data[0:2000],target[0:2000],i)


    plt.figure("Precision")
    plt.plot(precision)
    a.write_precision("precision1.csv",precision)


    precision2=np.zeros(len(A)-1)
    for i in range(len(A)-2,-1,-1):
        A,phi_tilde=a.left_sweep(A,phi_tilde,data[0:2000],target[0:2000],i)
        result2=a.test2(A,data[6000:7000],target[6000:7000],i)
        precision2[i]=sum(np.array(result2)==np.array(test_target))*1.0/len(test_target)

    plt.figure("Precision")
    plt.plot(precision2)

    a.write_precision("precision2.csv",precision2)

    precision3=np.zeros(len(A)-1)
    for i in range(len(A)-1):
        result2=a.test2(A,data[6000:7000],target[6000:7000],i)
        precision3[i]=sum(np.array(result2)==np.array(test_target))*1.0/len(test_target)
        A,phi_tilde=a.right_sweep(A,phi_tilde,data[0:2000],target[0:2000],i)


    plt.figure("Precision")
    plt.plot(precision3)
    plt.show()
    a.write_precision("precision3.csv",precision3)




