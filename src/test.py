from nltk.corpus import treebank
from nltk import treetransforms
from nltk import induce_pcfg
from nltk.parse import pchart
import nltk
import numpy as np
from numpy import mat
from gen_hmm_para import Hmm
from matrixop import read_matrix
import matplotlib.pyplot as plt
import matplotlib

"""importing the matrices from the files"""
seq=read_matrix("seq")
OM=read_matrix("om")
TPM=read_matrix("tpm")
PI=read_matrix("pi")
new_seq=read_matrix("new_seq")
'''instantiating the hmm object'''
hmm=Hmm(2,2,TPM,OM,PI)


leng=range(100)
error={}
#learning the hmm from a long sequance
hmm.learn_hmm(seq,2)


#finds the original probability form the model using forward backward algo and estimated prob from the hmm learned using spectral algorithms
for j in range(100):
    for i in leng:
        print j*100,'\t',j*100+i+1
        est=hmm.hmm_seq_prob(new_seq[:,j*100:j*100+i+1])
        org=hmm.prb_hmm(new_seq[:,j*100:j*100+i+1])
        print"------------------------------------------------------------------"
        if(j==0): 
            error[i]=(abs(org-est)/org)*100
        else:
            error[i]=error[i]+(abs(org-est)/org)*100
        

x=[]
y=[]
for i in error.keys():
    x.extend([i])
    y.extend(error[i]/100)
#plot the resulting error vs sequence length 
    
fig=plt.figure()
plt.plot(x,y,'ro')
plt.xlabel('length of the sequance')
plt.ylabel('mean error')
fig.savefig('test.jpg')
plt.show()



'''print hmm.B_inf
print hmm.B_one
print hmm.B_x
'''
