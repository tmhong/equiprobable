"""Check the boost-free hardware formulas against lib/angles.py on massless random events."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.angles import production_angles, decay_angles

rng = np.random.default_rng(1)
n = 20000
def massless(n):
    pt = rng.uniform(20, 300, n); eta = rng.uniform(-2.5, 2.5, n); phi = rng.uniform(-np.pi, np.pi, n)
    return np.stack([pt*np.cos(phi), pt*np.sin(phi), pt*np.sinh(eta), pt*np.cosh(eta)], -1)
def m2(a, b): return 2*(a[:,3]*b[:,3] - np.sum(a[:,:3]*b[:,:3], -1))
lep, nu, bl, q1, q2, bh = (massless(n) for _ in range(6))

def top_angles_hw(f, o, b):
    t = f + o + b
    P2 = np.sum(t[:,:3]**2, -1); rP = 1/np.sqrt(P2)
    pT2 = t[:,0]**2 + t[:,1]**2; rpT = 1/np.sqrt(pT2)
    z = t[:,:3]*rP[:,None]
    y = np.stack([-t[:,1]*rpT, t[:,0]*rpT, 0*rpT], -1)
    x = np.stack([y[:,1]*z[:,2], -y[:,0]*z[:,2], y[:,0]*z[:,1]-y[:,1]*z[:,0]], -1)
    mt2 = m2(f,o) + m2(f,b) + m2(o,b)
    rmt = 1/np.sqrt(mt2)
    gam = t[:,3]*rmt; bgam = P2*rP*rmt
    bh_ = b[:,:3]/b[:,3:4]; fh = f[:,:3]/f[:,3:4]
    cb, bx, by = np.sum(bh_*z,-1), np.sum(bh_*x,-1), np.sum(bh_*y,-1)
    cf, fx, fy = np.sum(fh*z,-1), np.sum(fh*x,-1), np.sum(fh*y,-1)
    nb = bgam - gam*cb; db = gam - bgam*cb
    cos_th = nb/db
    phi = np.arctan2(-by, -bx)
    cos_thp = (m2(f,b) - m2(o,b))/(m2(f,b) + m2(o,b))
    gz = gam*cf - bgam
    Yp = db*(by*fx - bx*fy)
    Xp = -nb*(bx*fx + by*fy) - (bx**2 + by**2)*gz
    phip = np.arctan2(Yp, Xp)
    return cos_th, phi, cos_thp, phip

def prod_hw(objs):
    T = sum(objs)
    tl = objs[0] + objs[1] + objs[2]
    MT2 = sum(m2(objs[i], objs[j]) for i in range(6) for j in range(i+1, 6))
    MT = np.sqrt(MT2)
    gam = T[:,3]/MT; beta = T[:,:3]/T[:,3:4]
    v = tl[:,:3]/tl[:,3:4]
    bv = np.sum(beta*v, -1)
    g = gam*gam/(gam+1)
    k = gam - g*bv
    u = v - k[:,None]*beta
    cosT = u[:,2]/np.sqrt(np.sum(u**2,-1))
    Y = 0.5*(np.log(T[:,3]+T[:,2]) - np.log(T[:,3]-T[:,2]))
    return Y, cosT

Y_r, cT_r = production_angles(lep+nu+bl, q1+q2+bh)
Y_h, cT_h = prod_hw([lep, nu, bl, q1, q2, bh])
print("Y", np.max(abs(Y_r-Y_h)), "cosTheta", np.max(abs(cT_r-cT_h)))
for name, (f, o, b) in {"lep": (lep, nu, bl), "had": (q2, q1, bh)}.items():
    ref = decay_angles(f+o+b, f+o, f)
    hw = top_angles_hw(f, o, b)
    d = [np.max(np.abs(r - h)) if i % 2 == 0 else np.max(np.abs((r - h + np.pi) % (2*np.pi) - np.pi)) for i, (r, h) in enumerate(zip(ref, hw))]
    print(name, "max |diff| cos_theta %.2e phi %.2e cos_theta' %.2e phi' %.2e" % tuple(d))
