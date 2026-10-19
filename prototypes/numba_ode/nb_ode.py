import numpy as np
from numba import njit

# Dormand-Prince 5(4) coefficients
c2,c3,c4,c5=1/5,3/10,4/5,8/9
a21=1/5
a31,a32=3/40,9/40
a41,a42,a43=44/45,-56/15,32/9
a51,a52,a53,a54=19372/6561,-25360/2187,64448/6561,-212/729
a61,a62,a63,a64,a65=9017/3168,-355/33,46732/5247,49/176,-5103/18656
b1,b3,b4,b5,b6=35/384,500/1113,125/192,-2187/6784,11/84
e1,e3,e4,e5,e6,e7=71/57600,-71/16695,71/1920,-17253/339200,22/525,-1/40

@njit(cache=True)
def dopri5(rhs, p, t0, t1, y0, rtol, atol, grid):
    """Adaptive DOPRI5 from t0 to t1 (either direction). If grid is non-empty,
    steps land exactly on each grid point (ordered t0->t1) and the state is stored there."""
    n = y0.size
    sgn = 1.0 if t1 > t0 else -1.0
    t = t0; y = y0.copy()
    h = sgn*abs(t1-t0)*1e-4
    out = np.empty((grid.size, n))
    gi = 0
    while gi < grid.size and (grid[gi]-t)*sgn <= 0.0:
        out[gi] = y; gi += 1
    k1 = rhs(t, y, p)
    nsteps = 0
    while (t1 - t)*sgn > 1e-15:
        target = t1 if gi >= grid.size else grid[gi]
        if (t + h - target)*sgn > 0.0:
            h = target - t
        k2 = rhs(t+c2*h, y+h*(a21*k1), p)
        k3 = rhs(t+c3*h, y+h*(a31*k1+a32*k2), p)
        k4 = rhs(t+c4*h, y+h*(a41*k1+a42*k2+a43*k3), p)
        k5 = rhs(t+c5*h, y+h*(a51*k1+a52*k2+a53*k3+a54*k4), p)
        k6 = rhs(t+h, y+h*(a61*k1+a62*k2+a63*k3+a64*k4+a65*k5), p)
        ynew = y+h*(b1*k1+b3*k3+b4*k4+b5*k5+b6*k6)
        k7 = rhs(t+h, ynew, p)
        err = h*(e1*k1+e3*k3+e4*k4+e5*k5+e6*k6+e7*k7)
        sc = atol + rtol*np.maximum(np.abs(y), np.abs(ynew))
        en = np.sqrt(np.mean((err/sc)**2))
        if en <= 1.0:
            t = t + h; y = ynew; k1 = k7; nsteps += 1
            if gi < grid.size and abs(t-grid[gi]) < 1e-14*max(1.0,abs(t)):
                t = grid[gi]; out[gi] = y; gi += 1
            fac = 5.0 if en == 0.0 else min(5.0, 0.9*en**-0.2)
        else:
            fac = max(0.2, 0.9*en**-0.2)
        h = h*fac
    return y, out, nsteps

@njit(cache=True)
def axial_rhs(u, y, p):
    # p = [w, zh, Q, mug, g5, ...], chi grid passed via p tail: [n, ugrid..., chigrid...]
    w, zh, Q, mug, g5 = p[0], p[1], p[2], p[3], p[4]
    n = int(p[5])
    ug = p[6:6+n]; cg = p[6+n:6+2*n]
    f = 1-(1+Q*Q)*u**4+Q*Q*u**6
    fp = -4*(1+Q*Q)*u**3+6*Q*Q*u**5
    phip = 2*u*mug*mug*zh*zh
    chi = np.interp(u, ug, cg)
    damp = 1/u - fp/f + phip
    pot = w*w*zh*zh/(f*f) - g5*g5*chi*chi/(u*u*f)
    d = np.empty(4)
    d[0] = y[1]; d[1] = y[1]*damp - y[0]*pot
    d[2] = y[3]; d[3] = y[3]*damp - y[2]*pot
    return d

@njit(cache=True)
def chiral_rhs(u, y, p):
    v3, v4, lambda1, mu_g, zh, q = p[0], p[1], p[2], p[3], p[4], p[5]
    Q = q*zh**3
    phi = (mu_g*zh*u)**2
    phip = 2*u*(zh*mu_g)**2
    f = 1 - (1+Q**2)*u**4 + Q**2*u**6
    fp = -4*(1+Q**2)*u**3 + 6*Q**2*u**5
    chi, chip = y[0], y[1]
    d = np.empty(2)
    d[0] = chip
    d[1] = (3/u-fp/f+phip)*chip - (3*chi+lambda1*phi*chi-3*v3*chi**2-4*v4*chi**3)/(u**2*f)
    return d
