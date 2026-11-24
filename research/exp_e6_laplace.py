"""E6 -- physics sanity: Laplace secular torque of naive retarded gravity.

Circular equal-mass binary (G = m = d = 1). Under naive retardation the lagging
source gives a tangential acceleration a_t = (Gm/d^2) sin(th/2)/cos^2(th/2),
th = Omega*tau, tau = d cos(th/2)/c (stationary circular orbit), so the total
angular momentum grows at dL/dt = d*m*a_t. The engine's slope of L(t) on
[2 tau, 2 tau + 0.2] should approach it as beta = v/c -> 0 (the formula ignores
that the orbit itself spirals out at a rate ~ beta).

Aberration model, same stationary-circle argument: the extrapolated source
s = x_j(t-tau) + tau v_j(t-tau) has tangential offset r (sin th - th cos th)
~ r th^3/3, so dL/dt = d * m * a_t ~ G m^2 th^3 / (6 d)  (beta^3, not beta).
"""
import json, os, sys
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, binary, stream

def aberration_exact(Om, c, d=1.0):
    r = d / 2
    th = brentq(lambda th: th - Om * d * np.cos(th / 2) / c, 0.0, 1.0)
    sm = np.array([-r - r * np.cos(th) - r * th * np.sin(th), r * np.sin(th) - r * th * np.cos(th)])
    return d * sm[1] / np.linalg.norm(sm) ** 3, th ** 3 / 6


rows = []
for c in (8.0, 16.0, 32.0, 64.0):
    p = binary(c=c, retardation="aberration")
    exact, lead = aberration_exact(p.omega, c)
    s = stream(p, 0.05, 24, track_breaks=3)
    T, L = [], []
    while s.t_epoch < 0.6:
        w = s.step()
        T.append(w.t)
        L.append(np.cross(w.trajectory, w.velocity)[:, :, 2].sum(1))
    t, Lt = np.concatenate(T), np.concatenate(L)
    m = (t >= 3 / c) & (t <= 3 / c + 0.2)
    slope = float(np.polyfit(t[m], Lt[m], 1)[0])
    rows.append(dict(model="aberration", c=c, analytic=float(exact), leading=float(lead),
                     engine=slope, rel_diff=abs(slope - exact) / exact))
    print(f"aberration c={c:4.0f} exact={exact:.4e} th^3/6={lead:.4e} engine={slope:.4e} "
          f"rel.diff={abs(slope-exact)/exact:.2%}")

for c in (8.0, 16.0, 32.0, 64.0):
    p = binary(c=c, retardation="naive")
    Om = p.omega
    th = brentq(lambda th: th - Om * np.cos(th / 2) / c, 0.0, 1.0)
    analytic = np.sin(th / 2) / np.cos(th / 2) ** 2
    s = stream(p, 0.05, 24, track_breaks=2)
    T, L = [], []
    while s.t_epoch < 0.6:
        w = s.step()
        T.append(w.t)
        L.append(np.cross(w.trajectory, w.velocity)[:, :, 2].sum(1))
    t, Lt = np.concatenate(T), np.concatenate(L)
    tau = 1.0 / c
    m = (t >= 2 * tau) & (t <= 2 * tau + 0.2)
    slope = float(np.polyfit(t[m], Lt[m], 1)[0])
    rows.append(dict(model="naive", c=c, beta=Om * 0.5 / c, analytic=float(analytic), engine=slope,
                     rel_diff=abs(slope - analytic) / analytic))
    print(f"c={c:4.0f} beta={Om*0.5/c:.4f} analytic={analytic:.6f} engine={slope:.6f} "
          f"rel.diff={abs(slope-analytic)/analytic:.3%}")
json.dump(rows, open(os.path.join(ROOT, "research/results/e6_laplace.json"), "w"), indent=1)
