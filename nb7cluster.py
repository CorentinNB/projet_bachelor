"""
Sweep en ε pour N=20 blocs — Erickson et al. (2011)
(version avec marqueurs '.-' sur tous les tracés)
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

EPS_LIST  = [0.02, 0.5, 2.0, 8.0, 12.0, 20.0]

CONFIG = {
    0.02: dict(T=1500, dt=20, ms=2.0,  rtol=1e-5, atol=1e-7, method='RK45'),
    0.5:  dict(T=1500, dt=20, ms=2.0,  rtol=1e-5, atol=1e-7, method='RK45'),
    2.0:  dict(T=75,   dt=1,  ms=0.5,  rtol=1e-5, atol=1e-7, method='Radau'),
    8.0:  dict(T=75,   dt=1,  ms=0.5,  rtol=1e-6, atol=1e-8, method='Radau'),
    12.0: dict(T=100,  dt=1,  ms=0.5,  rtol=1e-6, atol=1e-8, method='Radau'),
    20.0: dict(T=1000, dt=5,  ms=0.5,  rtol=1e-6, atol=1e-8, method='Radau'),
}
EPS0 = 1e-8

def make_y0():
    u0b   = -F_TILDE * GAMMA_LAM**2 / (XI * GAMMA_MU**2)
    x_bar = np.array([(j - 0.5) * 20.0 / N for j in range(1, N + 1)])
    u_init = u0b + np.exp(-((x_bar - 10.0)**2) / SIGMA**2)
    return np.concatenate([u_init, np.zeros(N), np.zeros(N)])

def make_rhs(eps):
    gm2 = GAMMA_MU**2; gl2 = GAMMA_LAM**2
    def rhs(t, y):
        u  = y[:N]; v  = y[N:2*N]; Th = y[2*N:]
        ul  = np.concatenate([[u[0]],  u[:-1]])
        ur  = np.concatenate([u[1:],   [u[-1]]])
        vp1 = np.maximum(v + 1.0, 1e-15); lv = np.log(vp1)
        dv  = gm2*(ul-2*u+ur) - gl2*u - (gm2/XI)*(F_TILDE+Th+lv)
        dTh = -vp1*(Th + (1.+eps)*lv)
        return np.concatenate([v, dv, dTh])
    return rhs

def _ivp(rhs, t0, t1, y, ms, rtol, atol, method='auto'):
    if method == 'auto':
        v_max = np.abs(y[N:2*N]).max()
        method = 'Radau' if v_max > 0.5 else 'RK45'
    return solve_ivp(rhs, [t0, t1], y,
                     method=method, rtol=rtol, atol=atol,
                     max_step=ms, dense_output=False)

def simulate(eps):
    cfg = CONFIG[eps]
    T, dt, ms, rtol, atol = cfg['T'], cfg['dt'], cfg['ms'], cfg['rtol'], cfg['atol']
    rhs   = make_rhs(eps)
    c_idx = N // 2

    y0  = make_y0()
    y0p = y0.copy(); y0p[0] += EPS0

    y = y0.copy(); yp = y0p.copy()
    t_cur = 0.0; log_sum = 0.0

    t_traj=[]; v_c=[]; u_c=[]; Th_c=[]
    t_ly=[]; Lambda=[]
    t_wall = time.time()

    while t_cur < T - 1e-10:
        t_end = min(t_cur + dt, T)
        method_cfg = cfg['method']
        sol  = _ivp(rhs, t_cur, t_end, y,  ms, rtol, atol, method_cfg)
        solp = _ivp(rhs, t_cur, t_end, yp, ms, rtol, atol, method_cfg)

        if sol.status != 0 or solp.status != 0:
            print(f'    [WARN] solver failed at t={t_cur:.1f}')
            break

        y  = sol.y[:, -1]
        yp = solp.y[:, -1]

        # Stocker trajectoire — décimation pour garder ~8 pts/chunk
        stride = max(1, sol.t.size // 8)
        for k in range(0, sol.t.size, stride):
            t_traj.append(sol.t[k])
            v_c.append(sol.y[N + c_idx, k])
            u_c.append(sol.y[c_idx, k])
            Th_c.append(sol.y[2*N + c_idx, k])

        diff = yp - y; nd = np.linalg.norm(diff)
        if nd > 0:
            log_sum += np.log(nd / EPS0)
            yp = y + diff * (EPS0 / nd)

        t_cur = t_end
        if t_cur > 0:
            t_ly.append(t_cur)
            Lambda.append(log_sum / t_cur)

    wall = time.time() - t_wall
    L_final = Lambda[-1] if Lambda else 0.

    half  = len(Lambda) // 2
    t_h   = np.array(t_ly[half:])
    tL_h  = t_h * np.array(Lambda[half:])
    slope = np.polyfit(t_h, tL_h, 1)[0] if len(t_h) > 4 else 0.
    is_chaos = (slope > 0.005) and (L_final > 0.003)

    print(f'  ε={eps:5.2f}  T={T}  wall={wall:.0f}s  '
          f'npts_traj={len(t_traj)}  npts_lya={len(t_ly)}  '
          f'Λ={L_final:.5f}  → {"CHAOS" if is_chaos else "périodique"}')

    return (np.array(t_traj), np.array(v_c), np.array(u_c), np.array(Th_c),
            np.array(t_ly),   np.array(Lambda), is_chaos, L_final)

def plot_sweep(results):
    n_eps  = len(EPS_LIST)
    cmap   = plt.cm.plasma
    colors = [cmap(i / (n_eps - 1)) for i in range(n_eps)]

    fig, axes = plt.subplots(n_eps, 4, figsize=(18, 3.2 * n_eps))
    fig.suptitle(
        r'N=20 blocs — sweep en $\varepsilon$   '
        r'[$\gamma_\mu=0.5,\ \gamma_\lambda=\sqrt{0.2},\ \xi=0.5,\ \tilde{f}=3.2$]'
        '\nBloc central — régime permanent (2ème moitié)\n'
        r'$\Lambda(t)$ : méthode deux trajectoires (Benettin 1980,  $\varepsilon_0=10^{-8}$)',
        fontsize=10, fontweight='bold'
    )

    col_titles = [
        r'Vitesse relative $\bar{v}_c$',
        r'Position relative $\bar{u}_c$',
        r'$\Lambda(t) = \frac{1}{t}\sum\ln\frac{\|\delta_i\|}{\varepsilon_0}$',
        r'État $\bar{\Theta}_c$',
    ]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9)

    for row, (eps, col) in enumerate(zip(EPS_LIST, colors)):
        (t_traj, v_c, u_c, Th_c,
         t_ly, Lambda, is_chaos, L_final) = results[eps]

        T      = CONFIG[eps]['T']
        t_half = T / 2
        mask_t = t_traj >= t_half
        mask_l = t_ly   >= t_half

        t_p  = t_traj[mask_t]
        v_p  = v_c[mask_t]; u_p = u_c[mask_t]; Th_p = Th_c[mask_t]
        t_L  = t_ly[mask_l]; L_p = Lambda[mask_l]

        n_traj = mask_t.sum()
        n_lya  = mask_l.sum()

        regime = ('CHAOS  Λ→{:.4f}'.format(L_final) if is_chaos
                  else 'périodique  Λ→0  ({:.4f})'.format(L_final))
        row_lbl = (rf'$\varepsilon={eps}$' + '\n' + regime
                   + f'\n({n_traj} pts traj, {n_lya} pts Λ)')

        MS = 3  # taille des marqueurs

        def ax(c): return axes[row, c]

        # ── col 0 : vitesse ──────────────────────────────────────────────────
        ax(0).plot(t_p, v_p, '.-', color=col, lw=0.6, markersize=MS)
        ax(0).axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)
        ax(0).set_ylabel(row_lbl, fontsize=8, rotation=0,
                         labelpad=110, va='center', color=col)

        # ── col 1 : position ─────────────────────────────────────────────────
        ax(1).plot(t_p, u_p, '.-', color=col, lw=0.6, markersize=MS)

        # ── col 2 : Lyapunov ─────────────────────────────────────────────────
        if len(t_L) > 1:
            ax(2).plot(t_L, L_p, '.-', color=col, lw=1.0, markersize=MS)
            ax(2).axhline(0, color='black', lw=0.8, ls='--')
            ax(2).fill_between(t_L, 0, L_p, where=(L_p > 0),
                               color='red',   alpha=0.15)
            ax(2).fill_between(t_L, 0, L_p, where=(L_p <= 0),
                               color='green', alpha=0.15)
            ax(2).axhline(L_final, color=col, lw=0.7, ls=':', alpha=0.8)
        ax(2).set_ylabel(r'$\Lambda$', fontsize=8)

        # ── col 3 : état θ ───────────────────────────────────────────────────
        ax(3).plot(t_p, Th_p, '.-', color=col, lw=0.6, markersize=MS)
        ax(3).axhline(0, color='gray', lw=0.5, ls='--', alpha=0.4)

        for c in range(4):
            axes[row, c].grid(True, alpha=0.2)
            axes[row, c].tick_params(labelsize=7)
            if row == n_eps - 1:
                axes[row, c].set_xlabel(r'$\bar{t}$', fontsize=8)

        del t_traj, v_c, u_c, Th_c, t_ly, Lambda
        gc.collect()

    plt.tight_layout()
    out = 'bk_eps_sweep_N20.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nFigure → {out}')

if __name__ == '__main__':
    print('Sweep ε — N=20 blocs  (Benettin 1980)\n')
    results = {}
    for eps in EPS_LIST:
        print(f'ε = {eps}')
        results[eps] = simulate(eps)
    plot_sweep(results)