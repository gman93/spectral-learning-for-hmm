from nltk.corpus import treebank
from nltk import treetransforms
from nltk import induce_pcfg
from nltk.parse import pchart
import nltk
import numpy as np
from numpy import mat
from gen_hmm_para import Hmm
from matrixop import read_matrix


seq=read_matrix("seq")
OM=read_matrix("om")
TPM=read_matrix("tpm")
PI=read_matrix("pi")
new_seq=read_matrix("new_seq")
hmm=Hmm(5,10,TPM,OM,PI)
hmm.learn_hmm(seq,5)

hmm.hmm_seq_prob(new_seq)
hmm.prb_hmm(new_seq)


'''print hmm.B_inf
print hmm.B_one
print hmm.B_x
'''
