import numpy as np
from numpy import mat
from gen_hmm_para import Hmm
from matrixop import read_matrix
import matplotlib.pyplot as plt
import  matplotlib

"""importing the matrices from the files"""
seq1=read_matrix("seq1")
OM1=read_matrix("om")
TPM1=read_matrix("tpm")
PI1=read_matrix("pi")
seq3=read_matrix("both_seq")
seq2=read_matrix("seq2")
OM2=read_matrix("om2")
TPM2=read_matrix("tpm2")
PI2=read_matrix("pi2")

new_seq=read_matrix("new_seq")
'''instantiating the hmm object'''

hmm1=Hmm(2,2,TPM1,OM1,PI1)
hmm2=Hmm(2,2,TPM2,OM2,PI2)
hmm3=Hmm(2,2)

leng=range(100)
error={}
#learning the hmm from a long sequence
hmm1.learn_hmm(seq1,2)
hmm2.learn_hmm(seq2,2)
hmm3.learn_hmm(seq3,2)
prob_vals=np.zeros((100,100),dtype=np.float128)

for i in range(100):
    for j in range(100):
        org1=1
        org2=1
        est1=1
        est2=1
        if(i+1<50):
            print "seq1=",j*100,"-",j*100+i+1
            est1=hmm1.hmm_seq_prob(new_seq[:,j*100:j*100+i+1])
            org1=hmm1.prb_hmm(new_seq[:,j*100:j*100+i+1])
        else:
            print "seq1=",j*100,"-",j*100+50
            print "seq2=",j*100+50,"-",j*100+i+1
            est1=hmm1.hmm_seq_prob(new_seq[:,j*100:j*100+50])
            org1=hmm1.prb_hmm(new_seq[:,j*100:j*100+50])
            
            est2=hmm2.hmm_seq_prob(new_seq[:,j*100+50:j*100+i+2])
            org2=hmm2.prb_hmm(new_seq[:,j*100+50:j*100+i+2])
            
        prob_vals[i,j]=org1*org2

        print"------------------------------------------------------------------"
        if(j==0): 
            print (abs((org1*org2)-(est1*est2))/(org1*org2))*100
            error[i]=((abs((org1*org2)-(est1*est2))/(org1*org2))*100)
        else:
            error[i]+=(abs(org1*org2-est1*est2))/(org1*org2)*100
prb=[]

for i in range(100):
    for j in range(100):
        if(j==0):
            prb.extend([prob_vals[i,j]*0.01])
        else:
            prb[i]+=prob_vals[i,j]*0.01
print prb
plt.plot(range(100),prb)
plt.show()            

x=[]
y=[]
for i in error.keys():
    x.extend([i])
    y.extend(error[i]/100)
#plot the resulting error vs sequence length     
fig=plt.figure();
plt.plot(x,y)
#finds the original probability form the model using forward backward algo and estimated prob from the hmm learned using spectral algorithms
print "end of ns hmm" 
for i in range(100):
    for j in range(100):
        
        est=hmm3.hmm_seq_prob(new_seq[:,j*100:j*100+i+1])
#         org=hmm3.prb_hmm(new_seq[:,j*100:j*100+200+1])
        print"------------------------------------------------------------------"
        if(j==0):
            error[i]=(abs(prob_vals[i,j]-est)/prob_vals[i,j])*100
        else:
            error[i]+=(abs(prob_vals[i,j]-est)/prob_vals[i,j])*100
          



x=[]
y=[]
for i in error.keys():
    x.extend([i])
    y.extend(error[i]/100)
#plot the resulting error vs sequence length     
plt.plot(x,y)
plt.xlabel('length of the sequance')
plt.ylabel('mean error')
fig.savefig('result.png')

plt.show()

