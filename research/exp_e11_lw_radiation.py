"""E11 -- physics check of the Lienard-Wiechert model: mutual radiation damping.

To O(1/c^2) the LW two-body problem is the conservative Darwin dynamics. The first
non-conservative term is the 1/c^3 cross term F_i = (2/3c^3) q_i q_j  d(a_j)/dt
(no self-force in the model). For the circular +-1 binary (unit masses, d = 1):
a_dot = -Omega^2 v, so the Darwin energy changes at

    dE/dt = (2/3c^3) sum_{i!=j} q_i q_j v_i . a_dot_j = -(4/3) Omega^2 v^2 / c^3
          = -1.3333 / c^3        (Omega^2 = 2, v^2 = 1/2)

i.e. exactly half of the system's Larmor dipole power (the other half would be
the self-force). Darwin energy (relativistic kinetic + Coulomb + Darwin term):
E = sum m c^2 (gamma - 1) + q1 q2 / r - q1 q2 /(2 c^2 r) [v1.v2 + (v1.n)(v2.n)].
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import ROOT, lw_binary, stream


def darwin_energy(R, V, c, q=(1.0, -1.0), m=(1.0, 1.0)):
    x1, x2, v1, v2 = R[:, 0], R[:, 1], V[:, 0], V[:, 1]
    rv = x1 - x2
    r = np.linalg.norm(rv, axis=1)
    n = rv / r[:, None]
    kin = sum(mi * c**2 * (1 / np.sqrt(1 - np.einsum("kd,kd->k", v, v) / c**2) - 1)
              for mi, v in zip(m, (v1, v2)))
    qq = q[0] * q[1]
    darwin = -qq / (2 * c**2 * r) * (np.einsum("kd,kd->k", v1, v2)
                                     + np.einsum("kd,kd->k", v1, n) * np.einsum("kd,kd->k", v2, n))
    return kin + qq / r + darwin


rows = []
for c in (8.0, 16.0, 32.0, 64.0):
    p = lw_binary(c=c)
    tau = 1.0 / c
    s = stream(p, 0.25, 24, track_breaks=200)
    T, E = [], []
    while s.t_epoch < 3.0:
        w = s.step()
        T.append(w.t)
        E.append(darwin_energy(w.trajectory, w.velocity, c))
    t, e = np.concatenate(T), np.concatenate(E)
    m = t >= 1.0                               # after the start-up transient (8 tau at c=8)
    slope = float(np.polyfit(t[m], e[m], 1)[0])
    pred = -4.0 / 3.0 / c**3
    rows.append(dict(c=c, slope=slope, predicted=pred, ratio=slope / pred))
    print(f"c={c:4.0f} dE_Darwin/dt engine={slope:+.4e} predicted={pred:+.4e} ratio={slope/pred:.4f}",
          flush=True)
json.dump(rows, open(os.path.join(ROOT, "research/results/e11_lw_radiation.json"), "w"), indent=1)
