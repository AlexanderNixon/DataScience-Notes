import numpy as np
from numpy.linalg import svd
from scipy.linalg import eig,norm

def main():
    # In this file the bidiagonalization and the qds algorithm are
    # implemented. The first part of the practical session can be seen
    # in the notebook file.
    # The following lines of code call the necessary functions to bidiagonalize A
    A = np.array([[2,3,5,1,5],[6,3,1,4,3],[8,2,1,9,1],[5,9,11,3,9]])

    print
    print 'Matrix A used:'
    print A
    H,B = bidiagonalize(A)
    print np.matrix.round(B,1)

    # Here the qds algoeithm is implemented to find the eigenvalues of B
    q_aux=np.zeros(len(H)+1)
    e_aux=np.zeros(len(H)+1)
    a=np.diagonal(B,offset=0)
    b=np.diagonal(B,offset=1)
    q=a**2
    e=b**2

    # The qds algorithm is called:
    q_aux = qds(A,q_aux,e_aux,q,e)
    print
    print 'Obtained values of the square of the singular values of B with the dqs algorithm:'
    print q_aux[0:len(A.T)]
    print
    print 'Squared singular values of B:'
    Ub,Sb,Vb = svd(B)
    print np.matrix.round(Sb**2)[::2]

def qds(A,q_aux,e_aux,q,e):
    while norm(e,np.inf)>1.e-14:
        for j in range(len(A.T)-1):
            q_aux[j] = q[j] + e[j] - e_aux[j-1]
            e_aux[j] = e[j] * (q[j+1]/q_aux[j])
        q_aux[len(A.T)] = q[len(A.T)] - e_aux[len(A.T)-1]
        q=q_aux
        e=e_aux
    return q_aux

def bidiagonalize(A):
    H=create_h(A)
    print
    print 'Creation of H:'
    print H
    print
    print 'Bidiagonal matrix:'
    return H,bidiag(H)

def create_h(A):
    zeros1=np.zeros((len(A.T),len(A.T)))
    zeros2=np.zeros((len(A),len(A)))
    H1=np.concatenate((zeros1,A.T),axis=1)
    H2=np.concatenate((A,zeros2.T),axis=1)
    return np.concatenate((H1,H2),axis=0)

def PA(bet,v,A):
    v=v.reshape(v.shape[0],-1)
    return A-np.dot(v,bet*np.dot(A.T,v).T)

def AP(bet,v,A):
    v=v.reshape(v.shape[0],-1)
    return A-np.dot(bet*np.dot(A,v),v.T)

def bidiag(A):
    m=A.shape[0]
    n=A.shape[1]
    for i in range(n):
        x=A[i:,i]
        v,bet=house(x)
        if i==0:
            A=PA(bet,v,A)
        else:
            A[i:,i:]=PA(bet,v,A[i:,i:])
        if i!=n-1:
            x=A[i,i+1:]
            v,bet=house(x)
            A[i:,i+1:]=AP(bet,v,A[i:,i+1:])
    return A

def house(x):
    n=x.shape[0]
    s=np.dot(x[1:n],x[1:n].T)
    v=np.zeros(n)
    v[0]=1
    for i in range (1,n):
        v[i]=x[i]
    if(s<1.e-14):
        bet=0
    else:
        mu=np.sqrt(x[0]*x[0]+s)
        if(x[0]<=0):
            v[0]=x[0]-mu
        else:
            v[0]=-s/(x[0]+mu);
        bet=2*v[0]*v[0]/(s+v[0]*v[0])
        v=v/v[0]
    return v,bet


if __name__=="__main__":
    main()
