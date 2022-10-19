"""
Created on Mon Jan 21 18:17:54 2019 

Please cite: AO paper by Emanuele Centurioni

@author: Zhe Liu
"""

#we use the notation N=n+ik n,k>=0
#First layer is the transparent (k=0) semi-infinite medium where 
#the light comes from, the incidence angle Phi.
#Last layer is the semi-infinite medium where the light comes out, 
#N can be complex here (k>=0).
#Thicknesses of these 2 layers of course are not used (can be 0)
#lam is a list of single wavelength: lam=[lam1, lam2, ...]
#the multilayer is defined by the list:
#structure=[[thickness1, N1, incoherence1, roughness1]], [thickness2, N2, ..]
#where N is a list: N1=[N1(lam1), N1(lam2), ...]
#incoherence is TRUE if the layer is incoherent
#roughness is in angstrom and refers to the top of the layer

import numpy as np 

def AdaptData(structure,lam,Fi):
    for i in range(len(structure)):
        structure[i][1]=np.array(structure[i][1])
    return structure,np.array(lam),structure[0][1]*np.sin(Fi)

def ComputeRT(structure,lam,Fi):#mixed coherent-incoherent
    structure,lam,NSinFi=AdaptData(structure,lam,Fi)
    Sp,Tp,Rp=globscatmatr(structure,lam,NSinFi,"p")#to treat all layer as coherent
    Ss,Ts,Rs=globscatmatr(structure,lam,NSinFi,"s")#use scatmatr instead
    T=(Tp+Ts)/2
    R=(Rp+Rs)/2
    return R,T

def ComputeA(structure, lam, Fi, j):#absorbance layer j
    return ComputeFlux(structure, lam, Fi, j, 0)-ComputeFlux(structure, lam, Fi, j, 1)

def ComputeFlux(structure, lam, Fi, j, x):
#compute Energy Flux in layer j at relative position x (0<=x<=1)
    structure,lam,NSinFi=AdaptData(structure,lam,Fi)
    Ep=EFLUX(structure, lam, NSinFi, j, x, "p")
    Es=EFLUX(structure, lam, NSinFi, j, x, "s")
    return (Ep+Es)/2

def ComputeRo(structure,lam,Fi):#
    structure,lam,NSinFi=AdaptData(structure,lam,Fi)
    S,Tp,Rp,m,rp=scatmatr(structure,lam,NSinFi,"p")
    S,Ts,Rs,m,rs=scatmatr(structure,lam,NSinFi,"s")
    ro=rp/rs
    delta=-np.log(ro).imag#because (n+ik) notation we must revert the sign of delta
    psi=np.arctan(abs(ro))
    N=structure[0][1].real
    den=(1+np.sin(2*psi)*np.cos(delta))**2
    e1=N**2*np.sin(Fi)**2*(np.tan(Fi)**2*(np.cos(2*psi)**2-np.sin(2*psi)**2*np.sin(delta)**2)/den+1)
    e2=N**2*np.sin(Fi)**2*np.tan(Fi)**2*np.sin(4*psi)*np.sin(delta)/den
    e=e1+e2*1j
    n=e**(0.5)
    test = delta < 0#convert -180 to 180 deg to 0 to 360 deg form
    delta = delta + test*2*np.pi
    return psi*180/np.pi,delta*180/np.pi,e1,e2,n.real,n.imag

def CosFi(N1SinFi1,N2):
#return cos(Fi2) from Snell Law N1 Sin(Fi1) = N2 Sin(Fi2) = ....
    return (1 - (N1SinFi1/N2)**2)**(0.5)

def scatmatr(structure,lam,NSinFi,pol):
#treat the multilayer as coherent and calculates the scattering matrix, T and R
    #calculates interfaces matrices
    intmatrix=[]
    for i in range(len(structure)-1):
        N1=structure[i][1]
        N2=structure[i+1][1]
        rough=structure[i+1][3]
        CosFi1=CosFi(NSinFi, N1)
        CosFi2=CosFi(NSinFi, N2)
        if pol == "p":#p polarized
            r=(N2*CosFi1-N1*CosFi2)/(N2*CosFi1+N1*CosFi2)
            t=(2*N1*CosFi1)/(N2*CosFi1+N1*CosFi2)
        else:#s polarized
            r=(N1*CosFi1-N2*CosFi2)/(N1*CosFi1+N2*CosFi2)
            t=(2*N1*CosFi1)/(N1*CosFi1+N2*CosFi2)
        #old version without rough interfaces
        #intmatrix.append(array([[1/t,r/t],[r/t,1/t]]))
        al=W((2*np.pi/lam)*(2*N1*CosFi1)*rough)
        be=W((2*np.pi/lam)*(2*N2*CosFi2)*rough)
        ga=W((2*np.pi/lam)*(N1*CosFi1-N2*CosFi2)*rough)
        gat=ga*t
        intmatrix.append(np.array([[1/gat,be*r/gat],[al*r/gat,(ga**2*(1-r**2)+al*be*r**2)/gat]]))
    #calculates phase shift matrices
    phamatrix=[]
    for i in range(1,len(structure)-1):
        N=structure[i][1]
        d=structure[i][0]
        beta=PhaseShift(d,N,NSinFi,lam)
        phamatrix.append(np.array([[np.exp(-1j*beta),lam*0],[lam*0,np.exp(1j*beta)]]))
    #creates the matrix list of the multilayer interface/layer/interface/layer etc..
    matrix=[]
    matrix.append(intmatrix[0])
    for i in range(len(structure) - 2):
        matrix.append(phamatrix[i])
        matrix.append(intmatrix[i+1])
    #calculates the matrix list product 
    S=np.identity(2)
    for i in range(len(matrix)-1,-1,-1):
        S=MATRIXMULT(matrix[i],S)
    #calculates R and T 
    r=S[1][0]/S[0][0]
    T=1/S[0][0]
    N1=structure[0][1]
    N2=structure[len(structure)-1][1]
    R=abs(r)**2#Reflectance
    if pol == "p":
        T=abs(T)**2*(np.conjugate(N2)*CosFi(NSinFi,N2)).real/((N1*CosFi(NSinFi,N1))).real#Transmittance
    else:
        T=abs(T)**2*(N2*CosFi(NSinFi,N2)).real/((N1*CosFi(NSinFi,N1))).real#Transmittance
    return S,T,R,matrix,r

def W1(q):#error function
    q=q.real
    c=-np.rough**2*q**2/2
    test = c < -50#avoid overflow 
    c=c*(1-test)-test*50
    return np.exp(c)

def W(q):
    q=q.real
    if max(q) == 0:
        c=1.0
    else:
        c=1.0
#        in_file=open("interface.txt","r")
#        in_file.readline()
#        interface=int(in_file.readline()[0:1])
#        in_file.close()
#        if interface == 1:#linear
#            c=np.sin(q*3**(0.5))/(q*3**(0.5))
#        elif interface == 2:#step
#            c=np.cos(q)
#        elif interface == 3:#exponential
#            c=1/(1+q**2/2)
#        elif interface == 4:#error function
#            c=np.exp(-q**2/2)
##    test = c < -50#avoid overflow 
##    c=c*(1-test)-test*50
    return c


def modmatr(S):
#calculates the modified matrix (real) associated to a classical coherent scattering matrix
    a=abs(S[0][0])**2
    b=-abs(S[0][1])**2
    c=abs(S[1][0])**2
    d=(abs(det(S))**2-abs(S[0][1]*S[1][0])**2)/a
    M=np.array([[a,b],[c,d]])
    return M.real

def incohescatmatr(structure,lam,NSinFi,pol):
#for a mixed coherent/incoherent multilayer calculates the modified scattering matrices
#the original multilayer is divided in coherent layers/incoherent layer/coherent layers ....
#example: c is for coeherent, i for incoherent, / is the interface
#original multilayer /1c/2c/3c/4i/5c/6i/ = 6 layers = 13 matrices
#i.e. interface matrix, phase shift matrix, interface matrix ......
#modified multilayer [/1c/2c/3c/][4i][/5c/][6i][/] = 5 matrices
#note that because in the example the last layer is incoherent the last interface has to be
#taken into account
    GS,start=[],0
    for i in range(len(structure)):
        if structure[i][2]==True or i==(len(structure)-1):
            #calculates the modified scattering matrix associated to a coherent multilayer
            S,T,R,Null1,Null2=scatmatr(structure[start:i+1],lam,NSinFi,pol)
            GS.append(modmatr(S))
            start=i
            if i != (len(structure)-1):
                #calculates the incoherent layer matrix
                N=structure[i][1]
                d=structure[i][0]
                beta=PhaseShift(d,N,NSinFi,lam)
                GS.append(np.array([[abs(np.exp(-1j*beta))**2,lam*0],[lam*0,abs(np.exp(+1j*beta))**2]]))
    return GS
   
def globscatmatr(structure,lam,NSinFi,pol):
#for a mixed coherent/incoherent multilayer return the modified scattering matrix, T and R
    GS=incohescatmatr(structure,lam,NSinFi,pol)
    #calculates the matrix list product 
    S=np.identity(2)
    for i in range(len(GS)-1,-1,-1):
        S=MATRIXMULT(GS[i],S)
    R=S[1][0]/S[0][0]
    T=1/S[0][0]
    N1=structure[0][1]
    N2=structure[len(structure)-1][1]
    if pol == "p":
        T=T*(np.conjugate(N2)*CosFi(NSinFi,N2)).real/((np.conjugate(N1)*CosFi(NSinFi,N1))).real
    else:
        T=T*(N2*CosFi(NSinFi,N2)).real/((N1*CosFi(NSinFi,N1))).real
    return S,T,R

def PhaseShift(d,N,NSinFi,lam):#compute phase shift
    beta = 2*np.pi*d*N*CosFi(NSinFi,N)/lam
    test = beta.imag < -50#avoid overflow
    b=test*beta.imag
    beta=beta-b*(1j)-(50j)*test
    test = beta.imag > 50#avoid overflow
    b=test*beta.imag
    beta=beta-b*(1j)+(50j)*test
    return beta

def det(S):#compute determinant
    return S[0][0]*S[1][1]-S[0][1]*S[1][0]
    
def MATRIXMULT(A,B):#I can't use standard matrixmultiply with array elements!
    a=A[0][0]*B[0][0]+A[0][1]*B[1][0]
    b=A[0][0]*B[0][1]+A[0][1]*B[1][1]
    c=A[1][0]*B[0][0]+A[1][1]*B[1][0]
    d=A[1][0]*B[0][1]+A[1][1]*B[1][1]
    return np.array([[a,b],[c,d]])

def MATRIXMULTVECT(A,B):#compute matrix vector product
    a=A[0][0]*B[0]+A[0][1]*B[1]
    b=A[1][0]*B[0]+A[1][1]*B[1]
    return np.array([a,b])

def EFLUX(structure, lam, NSinFi, j, x, pol):
#compute Energy Flux in layer j at relative position x (0<=x<=1)
    N0=structure[0][1]
    N=structure[j][1]
    d=structure[j][0]
    beta=PhaseShift(d,N,NSinFi,lam)*(1-x)
    cosFi=CosFi(NSinFi,N)
    cosFi0=CosFi(NSinFi,N0)
    #compute T
    S,T,R=globscatmatr(structure,lam,NSinFi,pol)
    #compute incoherent matrices
    GS= incohescatmatr(structure,lam,NSinFi,pol)
    #compute U+ and U- at the interfaces
    U=[]
    M=np.array([1/S[0][0],0*lam])
    U.append([M[0],M[1]])
    for i in range(len(GS)-1,-1,-1):
        M=MATRIXMULTVECT(GS[i],M)
        U.append([M[0],M[1]])
    U.reverse()
    #find in which incoherent matrix is layer j => c
    c=0
    test=False
    for i in range(j+1):
       if structure[i][2]!=test:
           c=c+1
           test=structure[i][2]
       elif structure[i][2]==True:
           c=c+2
    #U on the left of incoherent matrix c is U[c]
    #U on the right of incoherent matrix c is U[c+1]    
    if structure[j][2]==True:#layer j is incoherent
        #U at x (measured from left interface)
        L=(np.array([[abs(np.exp(-1j*beta))**2,lam*0],[lam*0,abs(np.exp(+1j*beta))**2]]))
        Ux=MATRIXMULTVECT(L,U[c+1])
        if pol == "p":
            EFlux=(Ux[0]-Ux[1])*(np.conjugate(N)*cosFi).real/((np.conjugate(N0)*cosFi0)).real
        else:
            EFlux=(Ux[0]-Ux[1])*(N*cosFi).real/((N0*cosFi0)).real
    else:#layer j is coherent
        #find last layer of coherent packet containing layer j
        e=j
        while True:
            e=e+1
            if structure[e][2]==True or e==(len(structure)-1):
                break
        #find first layer of coherent packet containing layer j            
        b=j
        while True:
            b=b-1
            if structure[b][2]==True or b==0:
                break
        S1,T,R,GS1,Null=scatmatr(structure[b:e+1],lam,NSinFi,pol)
        #compute Sj
        Sj=np.identity(2)
        for i in range(len(GS1)-1,(j-b)*2-1,-1):
            Sj=MATRIXMULT(GS1[i],Sj)
        L=(np.array([[np.exp(-1j*beta),lam*0],[lam*0,np.exp(+1j*beta)]]))
        #Light flux coming from left
        E=MATRIXMULTVECT(Sj,np.array([1/S1[0][0],lam*0])*(U[c][0])**(0.5))
        Ex=MATRIXMULTVECT(L,E)
        EFluxl=Poynting(Ex,N,N0,cosFi,cosFi0,pol)
        #Light flux coming from rigth
        E=MATRIXMULTVECT(Sj,np.array([-S1[0][1]/S1[0][0],lam/lam])*(U[c+1][1])**(0.5))        
        Ex=MATRIXMULTVECT(L,E)
        EFluxr=Poynting(Ex,N,N0,cosFi,cosFi0,pol)        
        #Total ligth flux
        EFlux=(EFluxl+EFluxr)
    return EFlux

def Poynting(Ex,N,N0,cosFi,cosFi0,pol):#compute energy flux
    if pol == "p":
        Flux=(abs(Ex[0])**2-abs(Ex[1])**2)*(np.conjugate(N)*cosFi).real-2*((Ex[0]*np.conjugate(Ex[1])).imag*(np.conjugate(N)*cosFi).imag)
    else:
        Flux=(abs(Ex[0])**2-abs(Ex[1])**2)*(N*cosFi).real-2*((Ex[0]*np.conjugate(Ex[1])).imag*(N*cosFi).imag)
    return Flux/(N0*cosFi0).real

def Alpha(structure,lam,Ti,a,Fi,i):#compute absorption coefficient
    Ti=np.array(Ti)
    a=np.array(a)
    structure[i][1]=np.array(structure[i][1])
    da=a/2
    for j in range(15):
        k=a*lam*1E-8/(4*np.pi)
        structure[i][1]=structure[i][1].real+k*(1j)
        R,T=ComputeRT(structure,lam,Fi)
        TiT=T/(1-R)
        test=TiT==Ti
        TiT=TiT*(1-test)#if Ti=TiT then set TiT to 0 
        sign=(TiT-Ti)/abs(TiT-Ti)#sign > 0 means alpha is too small
        sign=sign*(1-test)#if Ti=TiT then sign = 0 
        a=a+da*sign
        da=da/2
    return a

