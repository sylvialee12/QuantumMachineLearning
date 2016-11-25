__author__ = 'xlibb'
import numpy as np
def Ising(N,J,Jz,h):
    """
    :param N:
    :param J:
    :param Jz:
    :param h:
    :return:
    """
    sz=np.array([[1,0],[0,-1]])
    sp=np.array([[0,1],[0,0]])
    sm=np.array([[0,0],[1,0]])
    MPO=[[] for i in range(N)]
    MPO[1]=np.zeros([2,2,5,5])
    MPO[1][:,:,0,0]=np.eye(2)
    MPO[1][:,:,1,0]=sp
    MPO[1][:,:,2,0]=sm
    MPO[1][:,:,3,0]=sz
    MPO[1][:,:,4,0]=-h*sz
    MPO[1][:,:,4,1]=J/2*sm
    MPO[1][:,:,4,2]=J/2*sp
    MPO[1][:,:,4,3]=Jz*sz
    MPO[1][:,:,4,4]=np.eye(2)
    MPO[0]=MPO[1][:,:,4,:].reshape(2,2,1,5)
    MPO[N-1]=MPO[1][:,:,:,0].reshape(2,2,5,1)
    for i in range(2,N-1):
        MPO[i]=MPO[1]
    return MPO