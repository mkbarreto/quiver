import copy
import numpy as np
import numpy.random as rand
from matplotlib import pyplot as plt
from matplotlib import animation

N = 200
L = 100

#flock
r_v = 10
n_v = 18
eta = 0.1
lkahd = 5

T = 2*np.pi*rand.random((N,1))      #theta
v = .8
V = v*np.ones((N,1))                #velocity

Vmin = 6
Tmin = 2
Kp_T = .1
Kp_V = .05

S = np.zeros((N,1))

#predator
vP = [2.4]
n_p = int(.25*N)
tP = [2*np.pi*rand.random()]
Kp_Tp = .005
Kp_Vp = 0

fig, ax = plt.subplots(1,1)
Q = ax.quiver(L*rand.random((N,1)), L*rand.random((N,1)), np.cos(T), np.sin(T), scale=30,color='g')
Qs = ax.quiver(-L*np.ones((N,1)), -L*np.ones((N,1)), np.cos(T), np.cos(T), scale=30, color='k')
P = ax.quiver(L*rand.uniform(), L*rand.uniform(), np.cos(tP), np.sin(tP), scale=30, color='r')

ax.set_xlim(-.1*L, 1.1*L)
ax.set_ylim(-.1*L, 1.1*L)

def update_quiver(num, Q, T, V, P, tP, vP):
    """calculate new angle"""
    XY = copy.copy(Q.get_offsets())
    XYp = copy.copy(P.get_offsets())

    tempT = T
    for i in range(0,N):
        d = np.linalg.norm([XY[i,0]-XYp[0,0]-lkahd*vP[0]*np.cos(tP[0]),
                            XY[i,1]-XYp[0,1]-lkahd*vP[0]*np.sin(tP[0])])
        if S[i] == 0:
            """flock"""
            rXY = flock(T, V, XY, i)
            if (d<10) or (num_scared(XY,S,i) > 1):
                S[i] = 1
        else:
            """flee"""
            rXY = flee(XY, XYp, tP[0], vP[0], i)
            if d>25:
                S[i] = 0
#        rXY = flock(T, V, XY, i)

        """control towards desired position"""
        eVT = calc_err(XY[i],T[i],V[i],rXY)
        V[i] += Kp_V*eVT[0]
        tempT[i] += Kp_T*eVT[1]

        """apply physical limitations"""
        V[i] = min(max(V[i],.67*v),1.5*v)
        tempT[i] += eta*rand.uniform(-np.pi,np.pi)
    T = tempT

    """update predator"""
    rXYp = prey(XY,XYp)
    eVTp = calc_err(XYp[0],tP[0],vP[0],rXYp)
    vP[0] += Kp_Vp*eVTp[0]
    tP[0] += Kp_Tp*eVTp[1]

    """update positions"""
    Q.set_offsets(calc_offsets(Q,T,V,N,L))
    Q.set_UVC(np.cos(T),np.sin(T))

    Qs.set_offsets(Q.get_offsets()*S)
    Qs.set_UVC(np.cos(T*S),np.sin(T*S))

    P.set_offsets(calc_offsets(P,tP,vP,1,0))
    P.set_UVC(np.cos(tP),np.sin(tP))
    
    return

"""follow the n_v closest agents within distance r_v"""
def flock(T, V, XY, i):
    D = np.linalg.norm(XY-np.tile(XY[i],[N,1]), 2, 1)
    P = ((D < r_v)+.001)*(max(D)-D)
    Pi = rand.choice(N, min(n_v,np.sum(D < r_v)), False, P/np.sum(P))

    XYv = XY[Pi] + lkahd*V[Pi]*np.concatenate((np.cos(T[Pi]),np.sin(T[Pi])),1)
    rXY = np.mean(XYv,0)
    return rXY

"""move directly away from the predator"""
def flee(XY, XYp, tP, vP, i):
    rXY = np.zeros((2,1))
    rXY[0] = XY[i,0]-(XYp[0,0]+lkahd*vP*np.cos(tP))
    rXY[1] = XY[i,1]-(XYp[0,1]+lkahd*vP*np.sin(tP))
    rXY = 10*rXY/np.linalg.norm(rXY)
    rXY[0] = XY[i,0]+rXY[0]
    rXY[1] = XY[i,1]+rXY[1]
    return rXY

"""return number of scared birds w/in distance r_v of bird i (discouting self)"""
def num_scared(XY,S,i):
    D = np.linalg.norm(XY-np.tile(XY[i],[N,1]), 2, 1)
    C = [ind for ind,val in enumerate(D) if (val < r_v) and (ind != i)]
    return np.sum(S[C])

"""return average position of n_p birds closest to XYp"""
def prey(XY,XYp):
    N_p = np.argpartition(np.linalg.norm(XY-np.tile(XYp,[N,1]), 2, 1), n_p)[:n_p]
    return np.mean(XY[N_p],0)

"""calculate tangential and normal errors to within Vmin and Tmin"""
def calc_err(XY,T,V,rXY):
    XYv = [XY[0]+lkahd*V*np.cos(T),XY[1]+lkahd*V*np.sin(T)]
    eXY = [rXY[0] - XYv[0],rXY[1] - XYv[1]]
    eV = eXY[0]*np.cos(-T) - eXY[1]*np.sin(-T)
    eT = eXY[0]*np.sin(-T) + eXY[1]*np.cos(-T)

    vSgn = 2*(eV > 0)-1
    tSgn = 2*(eT > 0)-1
    eV = vSgn * (abs(eV) - Vmin) * (abs(eV)>Vmin)
    eT = tSgn * (abs(eT) - Tmin) * (abs(eT)>Tmin)

    return [eV,eT]

def calc_offsets(Q, T, V, N, L):
    XY = Q.get_offsets()
    for i in range(0,N):
        if L:
            XY[i,0] = (XY[i,0] + V[i]*np.cos(T[i]))%L
            XY[i,1] = (XY[i,1] + V[i]*np.sin(T[i]))%L
        else:
            XY[i,0] = (XY[i,0] + V[i]*np.cos(T[i]))
            XY[i,1] = (XY[i,1] + V[i]*np.sin(T[i]))
    return XY

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q,T,V,P,tP,vP),
                               interval=10, blit=False)
plt.show()