import numpy as np
from scipy import linalg
from matrixop import row_normalize
class Hmm:
    """A simple class to generate manupulate and acess simple hmms """
    def __init__(self,n,m,tpm=None,om=None,pi=None):
        """this creates a hmm objects """
        self.no_of_hidden_state=n
        self.no_of_observables=m
        self.TPM=tpm
        self.OM=om
        self.PI=pi
        self.B_one=None
        self.B_inf=None
        self.B_x=None
        
        
    
            
    def generate_hmm_para(self,no_of_hidden,no_of_obs): 
        """this function generate a random a Hmm"""
        TPM=np.random.rand((no_of_hidden,no_of_hidden),dtype = float64)
        OM=np.random.rand((no_of_obs,no_of_hidden),dtype = float64)
        PI=np.random.rand((1,no_of_obs),dtype = float64)
        
        TPM=row_normalize(TPM)
        OM=row_normalize(OM.T)
        PI=row_normalize(PI)
        
    def learn_hmm(self,seq):
        
        ''' this function learns the hmm parameters from a sequance of observations'''
        #print np.unique(seq.T)
        uni=np.zeros((1,10),dtype=np.float128)
        bi=np.zeros((10,10),dtype=np.float128)
        tri=np.zeros((10,10,10),dtype=np.float128)
        print tri.shape
        
        for i in range(seq.shape[1]-1):
        
            uni[0,int(seq[0,i]-1)]=1+int(uni[0,int(seq[0,i])-1])
            if(i<seq.shape[1]-1):
                bi[int(seq[0,i])-1,int(seq[0,i+1])-1]=1+bi[int(seq[0,i])-1,int(seq[0,i+1])-1]
            if(i+1<seq.shape[1]-1):
                tri[int(seq[0,i]-1),int(seq[0,i+2])-1,int(seq[0,i+1])-1]=1+tri[int(seq[0,i])-1,int(seq[0,i+2])-1,int(seq[0,i+1])-1]
        print tri.shape
        uni=np.divide(uni,seq.shape[1])
        bi=bi/seq.shape[1]
        tri=tri/seq.shape[1]
        
        [u,sing_val,v]=linalg.svd(bi)
        b_one=(u.T).dot(uni.T)
        
        
        b_inf=linalg.pinv(bi.T.dot(u)).dot(uni.T)
        b_x=np.zeros(tri.shape,dtype=np.float16)
        for i in range(tri.shape[0]):
            b_x[:,:,i]=u.T.dot(tri[:,:,i]).dot(linalg.pinv(u.T.dot(bi)))
        
        self.B_inf=b_inf
        self.B_one=b_one
        self.B_x=b_x
        print b_inf
        print "-------------------------------------------------------------------------"
        print b_one
        print "--------------------------------------------------------------------------"
        print b_x   
        


    def prb_hmm(self,seq):
        n=seq.shape[1]
        alpha=np.zeros((n,self.no_of_hidden_state),dtype=np.float128)
        for i in range(self.no_of_hidden_state):
            alpha[0,i]=self.OM[i,seq[0,i]-1]*(self.PI[0,i])
        for t in range(n-1):
            for j in range(self.no_of_hidden_state):
                z=0
                for k in range(self.no_of_hidden_state):
                    z=z+self.TPM[k,j]*alpha[t,k]
                       
                alpha[t+1,j]=z*self.OM[j,seq[0,t+1]-1]
        p=0
        for i in range(self.no_of_hidden_state):
            p=p+alpha[n-1,i]
        print "probability using forward backard algo",p
        return p    
                                
        
    def hmm_seq_prob(self,seq):
        "this function return the probability of a sequance according to the hmm "
        in_state=None
        for i in range(seq.shape[1]-1):
            print type(int(i))
            if in_state==None:
                in_state=self.B_x[:,:,seq[0,i]-1]
            else:
                in_state= self.B_x[:,:,seq[0,i]-1].dot(in_state)
        print self.B_inf.shape
        print self.B_one.shape
        print self.B_x.shape
        prob=self.B_inf.T.dot(in_state).dot(self.B_one)
        print "probability of the sequance is ",prob
        
        return prob
    
            
          