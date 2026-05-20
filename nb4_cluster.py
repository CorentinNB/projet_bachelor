"""
ε sweep for N=20 blocks — Erickson et al. (2011)
Two-trajectory Lyapunov (Benettin 1980) via augmented 6N system:
y and yp share identical step sizes → yp can never diverge mid-chunk.
Adaptive RK45/Radau switch based on max velocity at chunk start.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc, time

N         = 20
GAMMA_MU  = 0.5
GAMMA_LAM = np.sqrt(0.2)
XI        = 0.5
F_TILDE   = 3.2
SIGMA     = 1.0
EPS0      = 1e-8

EPS_LIST = [0.02, 0.5, 2.0, 8.0, 12.0, 20.0]

CONFIG = {
    0.02: dict(T=1500, dt=5.0, rtol=1e-6, atol=1e-8, v_thr=1.5, ms_rk=2.0,  ms_rd=0.2),
    0.5:  dict(T=1500, dt=5.0, rtol=1e-6, atol=1e-8, v_thr=1.5, ms_rk=2.0,  ms_rd=0.2),
    2.0:  dict(T=1500, dt=5.0, rtol=1e-5, atol=1e-7, v_thr=1.5, ms_rk=1.0,  ms_rd=0.05),
    8.0:  dict(T=1500, dt=5.0, rtol=1e-5, atol=1e-7, v_thr=1.5, ms_rk=1.0,  ms_rd=0.02),
    12.0: dict(T=1500, dt=5.0, rtol=1e-5, atol=1e-7, v_thr=1.5, ms_rk=1.0,  ms_rd=0.02),
    20.0: dict(T=1500, dt=5.0, rtol=1e-6, atol=1e-8, v_thr=1.5, ms_rk=1.0,  ms_rd=0.005),
}


# ── Model ────────────────────────────────────────────────────────────────────
def make_y0():
    u0b   = -F_TILDE * GAMMA_LAM**2 / (XI * GAMMA_MU**2)
    x_bar = np.array([(j - 0.5) * 20.0 / N for j in range(1, N + 1)])
    return np.concatenate([u0b + np.exp(-((x_bar - 10.0)**2) / SIGMA**2),
                           np.zeros(N), np.zeros(N)])


def make_rhs(eps):
    gm2 = GAMMA_MU**2;  gl2 = GAMMA_LAM**2
    def rhs(t, y):
        u = y[:N];  v = y[N:2*N];  Th = y[2*N:]
        ul  = np.concatenate([[u[0]], u[:-1]])
        ur  = np.concatenate([u[1:], [u[-1]]])
        vp1 = np.maximum(v + 1.0, 1e-15)
        lv  = np.log(vp1)
        dv  = gm2*(ul - 2*u + ur) - gl2*u - (gm2/XI)*(F_TILDE + Th + lv)
        dTh = -vp1 * (Th + (1. + eps)*lv)
        return np.concatenate([v, dv, dTh])
    return rhs


def make_rhs_augmented(eps):
    """6N augmented RHS: [y, yp] integrated simultaneously."""
    rhs = make_rhs(eps)
    def rhs_aug(t, state):
        return np.concatenate([rhs(t, state[:3*N]), rhs(t, state[3*N:])])
    return rhs_aug


# ── Adaptive step ─────────────────────────────────────────────────────────────
def _step_augmented(rhs_aug, t0, t1, state, cfg):
    """
    Integrate augmented 6N state one chunk.
    Method chosen based on max velocity of REFERENCE trajectory (first 3N).
    Both y and yp use identical steps → no mid-chunk divergence.
    """
    v_max = float(np.abs(state[N:2*N]).max())
    if v_max > cfg['v_thr']:
        method = 'Radau';  ms = cfg['ms_rd']
    else:
        method = 'RK45';   ms = cfg['ms_rk']

    return solve_ivp(rhs_aug, [t0, t1], state,
                     method=method,
                     rtol=cfg['rtol'], atol=cfg['atol'],
                     max_step=ms,
                     dense_output=False)


# ── Simulation ────────────────────────────────────────────────────────────────
def simulate(eps):
    cfg     = CONFIG[eps]
    T, dt   = cfg['T'], cfg['dt']
    rhs_aug = make_rhs_augmented(eps)
    c_idx   = N // 2

    y0    = make_y0()
    y0p   = y0.copy();  y0p[0] += EPS0
    state = np.concatenate([y0, y0p])

    t_cur   = 0.0
    log_sum = 0.0

    t_traj = []; v_c = []; u_c = []; Th_c = []
    t_ly   = []; Lambda = []
    t_wall = time.time()

    while t_cur < T - 1e-10:
        t_end = min(t_cur + dt, T)

        sol = _step_augmented(rhs_aug, t_cur, t_end, state, cfg)

        if sol.status != 0:
            # Retry with tighter settings
            cfg_tight = dict(cfg, ms_rd=cfg['ms_rd']*0.1,
                             ms_rk=cfg['ms_rk']*0.1,
                             rtol=cfg['rtol']*0.1, atol=cfg['atol']*0.1)
            sol = _step_augmented(rhs_aug, t_cur, t_end, state, cfg_tight)
            if sol.status != 0:
                print(f'    [WARN] solver failed at t={t_cur:.2f}')
                break

        state = sol.y[:, -1]
        y  = state[:3*N]
        yp = state[3*N:]

        # Store central block (decimated)
        stride = max(1, sol.t.size // 8)
        for k in range(0, sol.t.size, stride):
            t_traj.append(sol.t[k])
            v_c.append(sol.y[N + c_idx, k])
            u_c.append(sol.y[c_idx, k])
            Th_c.append(sol.y[2*N + c_idx, k])

        # Lyapunov renormalisation
        diff = yp - y
        nd   = np.linalg.norm(diff)
        if nd == 0.0 or not np.isfinite(nd):
            state[3*N:]    = y.copy()
            state[3*N]    += EPS0
        else:
            log_sum       += np.log(nd / EPS0)
            state[3*N:]    = y + diff * (EPS0 / nd)

        t_cur = t_end
        if t_cur > 0:
            t_ly.append(t_cur)
            Lambda.append(log_sum / t_cur)

    wall    = time.time() - t_wall
    L_final = Lambda[-1] if Lambda else 0.0

    half     = len(Lambda) // 2
    t_h      = np.array(t_ly[half:])
    tL_h     = t_h * np.array(Lambda[half:])
    slope    = np.polyfit(t_h, tL_h, 1)[0] if len(t_h) > 4 else 0.0
    is_chaos = (slope > 0.005) and (L_final > 0.003)

    print(f'  ε={eps:5.2f}  T={T}  wall={wall:.0f}s  '
          f'npts={len(t_traj)}  Λ={L_final:.5f}  '
          f'→ {"CHAOS" if is_chaos else "periodic"}')

    return (np.array(t_traj), np.array(v_c), np.array(u_c), np.array(Th_c),
            np.array(t_ly), np.array(Lambda), is_chaos, L_final)


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_sweep(results):
    n_eps  = len(EPS_LIST)
    colors = [plt.cm.plasma(i / (n_eps - 1)) for i in range(n_eps)]

    fig, axes = plt.subplots(n_eps, 4, figsize=(18, 3.2 * n_eps))
    fig.suptitle(
        r'N=20 blocks — sweep en $\varepsilon$  '
        r'[$\gamma_\mu=0.5,\ \gamma_\lambda=\sqrt{0.2},\ \xi=0.5,\ \tilde{f}=3.2$]'
        '\nCentral block — steady state (2nd half) | '
        r'$\Lambda(t)$: two-trajectory method (Benettin 1980, $\varepsilon_0=10^{-8}$)',
        fontsize=10, fontweight='bold')

    for c, title in enumerate([r'Speed $\bar{v}_c$', r'Position $\bar{u}_c$',
                                r'$\Lambda(t)$',      r'State $\bar{\Theta}_c$']):
        axes[0, c].set_title(title, fontsize=9)

    for row, (eps, col) in enumerate(zip(EPS_LIST, colors)):
        t_traj, v_c, u_c, Th_c, t_ly, Lambda, is_chaos, L_final = results[eps]
        T      = CONFIG[eps]['T']
        mask_t = t_traj >= T / 2
        mask_l = np.array(t_ly) >= T / 2
        t_p    = t_traj[mask_t]
        t_L    = np.array(t_ly)[mask_l]
        L_p    = np.array(Lambda)[mask_l]

        regime  = (f'CHAOS Λ→{L_final:.4f}' if is_chaos
                   else f'periodic Λ→0 ({L_final:.4f})')
        lbl     = rf'$\varepsilon={eps}$' + '\n' + regime

        MS = 3
        axes[row, 0].plot(t_p, v_c[mask_t],  '.-', color=col, lw=0.6, ms=MS)
        axes[row, 0].axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
        axes[row, 0].set_ylabel(lbl, fontsize=8, rotation=0,
                                labelpad=110, va='center', color=col)
        axes[row, 1].plot(t_p, u_c[mask_t],  '.-', color=col, lw=0.6, ms=MS)
        axes[row, 3].plot(t_p, Th_c[mask_t], '.-', color=col, lw=0.6, ms=MS)
        axes[row, 3].axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)

        if len(t_L) > 1:
            axes[row, 2].plot(t_L, L_p, '.-', color=col, lw=1.0, ms=MS)
            axes[row, 2].axhline(0, color='black', lw=0.8, ls='--')
            axes[row, 2].fill_between(t_L, 0, L_p, where=(L_p > 0),
                                      color='red', alpha=0.15)
            axes[row, 2].fill_between(t_L, 0, L_p, where=(L_p <= 0),
                                      color='green', alpha=0.15)
            axes[row, 2].axhline(L_final, color=col, lw=0.7, ls=':', alpha=0.8)
        axes[row, 2].set_ylabel(r'$\Lambda$', fontsize=8)

        for c in range(4):
            axes[row, c].grid(True, alpha=0.2)
            axes[row, c].tick_params(labelsize=7)
            if row == n_eps - 1:
                axes[row, c].set_xlabel(r'$\bar{t}$', fontsize=8)

        del t_traj, v_c, u_c, Th_c, t_ly, Lambda
        gc.collect()

    plt.tight_layout()
    out = 'bk_eps_sweep_N20.png'
    plt.savefig(out, dpi=140)
    plt.close(fig)
    print(f'\nFigure → {out}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Sweep ε — N=20 blocks (augmented 6N system, T=1500)\n')
    print('Note: ε=8, 12, 20 may be slow locally — use EPFL cluster.\n')
    results = {}
    for eps in EPS_LIST:
        print(f'ε = {eps}')
        results[eps] = simulate(eps)
    plot_sweep(results)