__author__ = 'xlibb'
import pickle
def loadnetwork(filename):
    with open(filename,'rb') as f:
        result=pickle.load(f)
    return result


if __name__=="__main__":
    trainpreci=loadnetwork("trainpreci.txt")
    trainlost=loadnetwork("trainlost.txt")
    print(trainlost)
    print(trainpreci)