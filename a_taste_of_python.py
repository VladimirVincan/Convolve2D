import numpy as np
import matplotlib.pyplot as plt


def discreteFT(fdata, N,fwd):
    X=[0]*(2*N)
    omega=0.0
    k, ki, kr, n = 0,0,0,0
    if(fwd):
        omega = 2.0*np.pi/N
    else:
        omega = -2.0*np.pi/n
    for k in range(0,N):
        kr = 2*k
        ki = 2*k+1
        X[kr]=0.0
        X[ki]=0.0
        for n in range(0,N):
            X[kr] += fdata[2*n]*np.cos(omega*n*k)+fdata[2*n+1]*np.sin(omega*n*k)
            X[ki] += fdata[2*n]*np.sin(omega*n*k)+fdata[2*n+1]*np.cos(omega*n*k)
    if(fwd):
        for k in range(0,N):
            X[2*k]/=N
            X[2*k+1] /= N
    return X


def fastFFT(fdata, N, fwd):
    omega, tempr, tempi, fscale= 0.0,0.0,0.0,0.0
    xtemp,cosine,sine,xr,xi=0.0,0.0,0.0,0.0,0.0
    i,j,k,n,m,M=0,0,0,0,0,0

    for i in range(0,N-1):
        if(i<j):
            tempr = fdata[2*i]
            tempi = fdata[2*i+1]
            fdata[2*i] = fdata[2*j]
            fdata[2*i+1]=fdata[2*j+1]
            fdata[2*j]=tempr
            fdata[2*j+1]=tempi
        k = N/2
        while(k <= j):
            j -=k
            k >>=1
        j+=k
        
    if(fwd):
        fscale = 1.0
    else:
        fscale = -1.0
        
    M = 2
    while(M<2*N):
        omega = fscale*2.0*np.pi/M
        sin = np.sin(omega)
        cos = np.cos(omega)-1.0
        xr = 1.0
        xi = 0.0
        for m in range(0,M-1,2):
            for i in range(m,2*N,M*2):
                j = i+m
                tempr = xr*fdata[j]-xi*fdata[j+1]
                tempi = xr*fdata[j+1]+xi*fdata[j]
                fdata[j]=fdata[i]-tempr
                fdata[j+1]=fdata[i+1]-tempi
                fdata[i]+=tempr
                fdata[i+1]+=tempi
            xtemp = xr
            xr = xr + xr*cos - xi*sin
            xi = xi+xtemp*sin+xi*cos
            M*=2
        if(fwd):
            for k in range(0,N):
                fdata[2*k]/=N
                fdata[2*k+1]/=N


def mainDFT():
    N = 64
    T = 2.0
    b = True
    fdata=[0]*(2*N)
    for i in range(0,N):
        fdata[2*i] = np.cos(4.0*np.pi*i*T/N)
        fdata[2*i+1] = 0.0
    Y=discreteFT(fdata,N,b)
    
    X = np.linspace(0.0,1.0/(2.0*T),N/2)
    yf = 2.0/N * np.abs(Y[0:int(N/2)])

    plt.plot(X,yf)
    plt.grid()
    plt.show()

    
def mainFFT():
    N = 64
    T = 1.0
    tn,fk = 0.0,0.0
    fdata = [0]*(2*N)
    for i in range(0,N):
        fdata[2*i] = np.cos(8.0*np.pi*i*T/N)+3*np.cos(14.0*np.pi*i*T/N)+ 2*np.cos(32.0*np.pi*i*T/N)
        fdata[2*i+1]=0.0
        
    y = discreteFT(fdata,N,True)
    fdata = (fdata[0:N])
    y = (y[0:N])
    x = np.linspace(0.0,1.0/(2.0*T),N)
    plt.plot(x,y)
    plt.grid()
    plt.show()


mainFFT ()
mainDFT()