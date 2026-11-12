"""
GENERATED FILE -- do not edit.  Written by derivations/derive_chiral_uv_series.py.

UV series of the chiral field through zhat^5, including the zhat^3 ln, zhat^4 ln
and zhat^5 ln terms, and its Jacobian with respect to (a, c).  z is zhat = mu_g z;
L = ln(zhat), i.e. the scale inside the logarithm is mu_g.
"""
import numpy as np


def uv_series(z, a, c, lam1, lam3, lam4, k4):
    """Return S = [chi, dchi/dzhat] and JS = dS/d(a, c) (2x2)."""
    L = np.log(z)
    t0 = -a
    t1 = a**2
    t2 = lam3*z
    t3 = t1*t2
    t4 = z**2
    t5 = lam4*t1
    t6 = lam3**2
    t7 = t1*t6
    t8 = lam1 - 4*t5 + 18*t7 - 2
    t9 = L*t8
    t10 = a*t9
    t11 = t4*(-2*c + t10)
    t12 = 20*a
    t13 = a**3
    t14 = t13*t6
    t15 = 7*lam1
    t16 = lam4*t13
    t17 = -a*t15 - 6*c + t12 - 99*t14 + 52*t16
    t18 = z**3
    t19 = lam3*t18
    t20 = a*t19*(3*t10 + t17)
    t21 = z**4
    t22 = 48*c
    t23 = 8*k4
    t24 = 16*lam1
    t25 = 8*lam1
    t26 = 3*lam1**2
    t27 = 96*c
    t28 = a**5
    t29 = lam3**4
    t30 = lam4**2
    t31 = lam4*t6
    t32 = lam1 - 12*t5 + 6*t7 - 6
    t33 = 4*t32
    t34 = a*t23 + a*t24 - a*t26 - c*t25 + 40*lam1*t14 + 48*lam1*t16 + t10*t33 - t12 - 104*t14 - 112*t16 - t22*t7 + t22 + t27*t5 + 1260*t28*t29 - 144*t28*t30 + 752*t28*t31
    t35 = a*t2
    t36 = t32*t4
    t37 = (1/16)*t4
    t38 = -2*lam4
    t39 = t38 + 9*t6
    t40 = t1*t39
    t41 = 4*t40
    t42 = L*t4*(t41 + t8)
    t43 = t13*t39
    t44 = t19*(12*L*t43 - a*(t15 - 156*t5 + 297*t7 - 20) + 6*t10 + t17)
    t45 = a**4
    t46 = t1*(t38 + t6)
    t47 = 16*L*t32*t40 + 192*a*c*lam4 - a*t27*t6 + 144*lam1*t5 + 120*lam1*t7 + t23 + t24 - t26 + 6300*t29*t45 - 720*t30*t45 + 3760*t31*t45 + t33*t9 + 48*t46*t9 - 336*t5 - 312*t7 - 20
    S = np.empty(2)
    JS = np.empty((2, 2))
    S[0] = z*(-t0 - 1/2*t11 - 1/3*t20 + (1/64)*t21*t34 - 3*t3)
    S[1] = -a*t37*t8*(16*t35 - t36 + 8) - t0 - 3/2*t11 - 4/3*t20 + (5/64)*t21*t34 - 6*t3
    JS[0, 0] = z*((1/64)*t21*t47 - 6*t35 - 1/2*t42 - 1/3*t44 + 1)
    JS[0, 1] = t18*(2*t35 - 1/8*t36 + 1)
    JS[1, 0] = (5/64)*t21*t47 - 12*t35 - t37*(64*t2*t43 + t25 + 32*t35*t8 - t36*t41 - t36*t8 - 12*t4*t46*t8 + 32*t40 - 32*t5 + 144*t7 - 16) - 3/2*t42 - 4/3*t44 + 1
    JS[1, 1] = t4*(8*t35 - 5/8*t36 + 3)
    return S, JS


def uv_log_coefficient(a, lam1, lam3, lam4):
    """Coefficient D of zhat^3 ln(zhat); it sets the scheme dependence of sigma."""
    return -1/2*a*(18*a**2*lam3**2 - 4*a**2*lam4 + lam1 - 2)
