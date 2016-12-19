import numpy as np
from numpy import random
from numpy import tensordot
import MPOIsing
from numpy import transpose
import matplotlib.pyplot as plt
from collections import namedtuple

class tree_TN2D(object):
	"""docstring for tree_TN2D"""
	def __init__(self, Nlayer,Dbond,d,Dout):
		super(tree_TN2D, self).__init__()
		self.Nlayer= 0
		self.Dbond=Dbond
		self.d=d
		self.Dout=Dout

	def initialize(self,Nx,Ny):
		
		self.Nlayer=np.log2(N).astype('int')
		W=[[] for i in range(self.Nlayer)]
		W[0]=[[random.rand(self.d,self.d,self.Dbond) for i in range(Nx//2)] for j in range(Ny//2) ]
		for i in range(1,self.Nlayer-1):
			pass
		