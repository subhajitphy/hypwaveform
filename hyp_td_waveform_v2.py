import numpy as np
from numpy import sin, cos, cosh, sqrt, sinh
import matplotlib.pyplot as plt
from constants import *
from rr_method1 import solve_rr
from getx_v2 import get_x, get_x_PN
from fn import rx, phitx, phiv, rtx
from hypmik3pn import get_u, get_u_v2
from eval_max import get_max, Fomg

from pycbc.types import TimeSeries


Mpc=1e6*pc


def get_hyp_waveform(M,q,et0,b,delta_t,inc,distance,order):
    ti=M/50
    eta=q/(1+q)**2
    time=M*tsun
    dis=M*dsun
    scale=distance/dis
    x0=get_x(et0,eta,b,3)[0]
    n0=x0**(3/2)
    tarr=np.arange(-ti,ti,delta_t)
    t_arr=tarr/time
    t_i=t_arr[0]
    t_f=t_arr[len(t_arr)-1]
    l_i=n0*t_i
    u_i=get_u(l_i,et0,eta,b,3)
    y0=[et0,n0,u_i]
    sol=solve_rr(eta,b,y0,t_i,t_f,t_arr)
    u_method1=sol[2]
    earr=sol[0]
    narr=sol[1]
    step=len(tarr)
    hp_arr=np.zeros(step)
    hx_arr=np.zeros(step)
    X=np.zeros(step)
    Y=np.zeros(step)
    for i in range(step):
        et=earr[i]
        u=u_method1[i]
        x=get_x(et,eta,b,order)[0]   
        phi=phiv(eta,et,u,x,order)
        r1=rx(eta,et,u,x,order)
        z=1/r1
        phit=phitx(eta,et,u,x,order)
        rt=rtx(eta,et,u,x,order)
        phi=phiv(eta,et,u,x,order)
        phi=phiv(eta,et,u,x,order)
        r1=rx(eta,et,u,x,order)
        X[i]=r1*cos(phi)
        Y[i]=r1*sin(phi)
        hp_arr[i]=(-eta*(sin(inc)**2*(z-r1**2*phit**2-rt**2)+(1+cos(inc)**2)*((z
        +r1**2*phit**2-rt**2)*cos(2*phi)+2*r1*rt*phit*sin(2*phi))))
        hx_arr[i]=(-2*eta*cos(inc)*((z+r1**2*phit**2-rt**2)*sin(2*phi)-2*r1*rt*phit*cos(2*phi)))
    Hp=TimeSeries(hp_arr/scale, delta_t=delta_t, epoch=-ti)
    Hx=TimeSeries(hx_arr/scale, delta_t=delta_t, epoch=-ti)
    #dimless_peak=get_max(eta,b,et0)
    #peak=dimless_peak/(2*np.pi*time)
    #return Hp-Hp[0],Hx-Hx[0], X, Y, peak
    return Hp-Hp[0],Hx-Hx[0]

    
