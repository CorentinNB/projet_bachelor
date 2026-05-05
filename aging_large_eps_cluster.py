"""
aging_large_eps_cluster.py
==========================
Comparaison Slip law vs Aging law pour eps=8 et eps=20, N=20 blocs.
À lancer sur le cluster EPFL (Linux, scipy récent, pas de bug Radau).

Pourquoi le cluster :
  - Aging law raide (resticking N=20 blocs couplés)
  - Radau implicite requis : LSODA bloque sur Windows, Radau OK sur Linux
  - Clip exp à 5 : réduit dTh_max de ~200k à ~1300 pour eps=8
                   → raideur gérable par Radau en temps raisonnable

Sorties :
  aging_eps8_eps20.png   — figure de comparaison (4 panneaux)
  aging_eps8_eps20.npz   — données brutes (rechargeable dans le notebook)

Usage :
  python aging_large_eps_cluster.py
  # ou sur SLURM :
  # sbatch --time=2:00:00 --mem=4G --wrap="python aging_large_eps_cluster.py"
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time, warnings, gc

# ── Paramètres — identiques au notebook ──────────────────────────────────────
N         = 20
GAMMA_MU  = 0.5
GAMMA_LAM = np.sqrt(0.2)
XI        = 0.5
F_TILDE   = 3.2
SIGMA     = 1.0
EPS0_IC   = 1e-8   # perturbation pour Lyapunov

# Formule originale make_y0 du notebook (ne pas modifier)
def make_y0():
    u0b   = -F_TILDE * GAMMA_LAM**2 / (XI * GAMMA_MU**2)
    x_bar = (np.arange(1, N+1) - 0.5) * 20.0 / N
    u_init = u0b + np.exp(-((x_bar - 10.0)**2) / SIGMA**2)
    return np.concatenate([u_init, np.zeros(N), np.zeros(N)])

# ── RHS ──────────────────────────────────────────────────────────────────────
def make_rhs_slip(eps):
    gm2 = GAMMA_MU**2; gl2 = GAMMA_LAM**2
    def rhs(t, y):
        u = y[:N]; v = y[N:2*N]; Th = y[2*N:]
        ul = np.r_[u[0], u[:-1]]; ur = np.r_[u[1:], u[-1]]
        vp1 = np.maximum(v + 1., 1e-15); lv = np.log(vp1)
        dv  = gm2*(ul-2*u+ur) - gl2*u - (gm2/XI)*(F_TILDE+Th+lv)
        dTh = -vp1 * (Th + (1.+eps)*lv)
        return np.concatenate([v, dv, dTh])
    return rhs

def make_rhs_aging(eps, clip=5.):
    """
    Aging law avec clip exp.
    clip=5 : exp(5)≈148 → dTh_max ≈ eps1*148.
    Physiquement sûr : en régime glissement (vp1=100), Th_ss = -eps1*ln(100)
    → -Th_ss/eps1 = ln(100)≈4.6 < 5. Le clip ne s'active que sur les
    transitoires hors-équilibre, pas sur la physique.
    """
    eps1 = 1. + eps; gm2 = GAMMA_MU**2; gl2 = GAMMA_LAM**2
    def rhs(t, y):
        u = y[:N]; v = y[N:2*N]; Th = y[2*N:]
        ul = np.r_[u[0], u[:-1]]; ur = np.r_[u[1:], u[-1]]
        vp1 = np.maximum(v + 1., 1e-15); lv = np.log(vp1)
        dv  = gm2*(ul-2*u+ur) - gl2*u - (gm2/XI)*(F_TILDE+Th+lv)
        dTh = eps1 * (np.exp(np.minimum(-Th/eps1, clip)) - vp1)
        return np.concatenate([v, dv, dTh])
    return rhs

# ── Config par eps ────────────────────────────────────────────────────────────
# eps=8  : T=75  identique à la slip law du notebook
# eps=20 : T=75  (slip law utilise T=1000 pour Lyapunov, ici on veut
#          juste la comparaison qualitative → T=75 suffit)
CONFIG = {
    8.0:  dict(T=75,  ms=0.30, rtol=1e-6, atol=1e-8),
    20.0: dict(T=75,  ms=0.20, rtol=1e-5, atol=1e-7),
}
EPS_LIST = [8.0, 20.0]

# ── Intégration ───────────────────────────────────────────────────────────────
def run(eps, law):
    cfg  = CONFIG[eps]
    T    = cfg['T']; ms = cfg['ms']
    rtol = cfg['rtol']; atol = cfg['atol']

    rhs = make_rhs_slip(eps) if law == 'slip' else make_rhs_aging(eps)
    y0  = make_y0()

    print(f"  eps={eps} {law:5s} T={T} Radau ms={ms}...", flush=True)
    t_wall = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sol = solve_ivp(rhs, [0, T], y0,
                        method='Radau',
                        rtol=rtol, atol=atol,
                        max_step=ms,
                        dense_output=False)

    elapsed = time.time() - t_wall
    ok = sol.t[-1] >= T * 0.99

    v_mid = sol.y[N + N//2, :]
    v_max = sol.y[N:2*N, :].max()

    print(f"    → {len(sol.t)} pts  {elapsed:.0f}s  ok={ok}  v_max={v_max:.1f}")
    if not ok:
        print(f"    [WARN] intégration incomplète : t_end={sol.t[-1]:.2f}")

    return sol.t, v_mid, elapsed

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Aging law cluster script — eps=8 et eps=20, N=20 blocs")
print("Méthode : Radau, clip exp=5")
print("=" * 60)

results = {}
t_total_start = time.time()

for eps in EPS_LIST:
    print(f"\nε = {eps}")
    for law in ['slip', 'aging']:
        t_arr, v_arr, elapsed = run(eps, law)
        results[(eps, law)] = (t_arr, v_arr)
        gc.collect()

total_elapsed = time.time() - t_total_start
print(f"\nTemps total : {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

# ── Sauvegarder les données brutes ───────────────────────────────────────────
save_dict = {}
for (eps, law), (t_arr, v_arr) in results.items():
    key = f"eps{eps}_law{law}"
    save_dict[f"{key}_t"] = t_arr
    save_dict[f"{key}_v"] = v_arr

np.savez('aging_eps8_eps20.npz', **save_dict)
print("Données → aging_eps8_eps20.npz")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(EPS_LIST), 2,
                         figsize=(13, 4.5 * len(EPS_LIST)),
                         sharex='row', sharey='row')

fig.suptitle(
    r'Slip law vs Aging law' + ' — N=20  [eps=8, eps=20]'
    r'\nRadau (cluster), clip exp a 5',
    fontsize=12, fontweight='bold')

axes[0, 0].set_title('Slip law (Ruina 1983)',    fontsize=10, fontweight='bold')
axes[0, 1].set_title('Aging law (Dieterich 1979)', fontsize=10, fontweight='bold')

COLORS = {'slip': '#1D9E75', 'aging': '#D85A30'}
CMAP   = [plt.cm.plasma(v) for v in [0.35, 0.75]]

for row, eps in enumerate(EPS_LIST):
    T = CONFIG[eps]['T']
    for col, law in enumerate(['slip', 'aging']):
        t_arr, v_arr = results[(eps, law)]
        mask = t_arr >= T / 2          # ignorer le transitoire
        col_c = COLORS[law]
        axes[row, col].plot(t_arr[mask], v_arr[mask], '.-',
                            color=col_c, lw=0.5, ms=1.5, alpha=0.85)
        axes[row, col].axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
        axes[row, col].grid(True, ls=':', alpha=0.35)
        axes[row, col].set_ylabel(
            rf'$\bar{{v}}_c$,  $\varepsilon={eps}$', fontsize=9)

for col in range(2):
    axes[-1, col].set_xlabel(r'$\bar{t}$', fontsize=9)

plt.tight_layout()
plt.savefig('aging_eps8_eps20.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("Figure → aging_eps8_eps20.png")
