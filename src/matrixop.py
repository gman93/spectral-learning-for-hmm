import numpy as np
def row_normalize(x):
    for i in range(x.shape(0)):
        x[i,:]=x[i,:]/x[i,:].sum()
    return x

def mean_normalize(mat):
    for i in range(mat.shape[1]):
        mat[:,i]=np.divide(mat[:,i],np.mean(mat[:,i]))
    return mat


def read_matrix(file_name,delimiter="\t"):
    
    """ this function returns a matrix stored in a file where the values are separated by a delimiter"""

    input_file=open(file_name)
    lines=input_file.readlines()
    i=0
    temp=lines[0]
    temp=temp.split(delimiter)
    matrix=np.zeros((len(lines),len(temp)),dtype=np.float128)
    
    for line in lines:
        s=line.split("\t")
        j=0
        for x in s:
            matrix[i,j]=float(x)
            j+=1
        i+=1
    return matrix


            