import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mcak_explore import main as run_mcak, lgM, run_mforce
import cgs_constants as cgs
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
from time import time


def get_Mforce_parameters(temperature, rho, lgt, workdir):
    """"""
    parameters = {"lgTmin": f"{np.log10(temperature):.3E}",
                  "lgTmax": f"{np.log10(temperature):.3E}",
                  "N_lgT": "1",
                  "lgDmin": f"{np.log10(rho):.5E}",
                  "lgDmax": f"{np.log10(rho):.5E}",
                  "N_lgD": "1",
                  "lgttmin": f"{lgt:.5E}",
                  "lgttmax": f"{lgt:.5E}",
                  "N_tt": "1",
                  "Ke_norm": "-10",
                  "ver": False,
                  "DIR": f"{workdir}/output"}
    return parameters


def mom_eq(r, M_t, v_crit, M_star, qbar, gamma_e):
    G = cgs.G
    GM = G * M_star * cgs.Msun
    Gamma_r = np.maximum(gamma_e * (1 + M_t), 1.01)

    # c1 = GM * (Gamma_r - 1)
    # gamma_r_func = interp1d(np.log10(r), np.log10(c1), kind="linear")
    # func = lambda r, v: 10**gamma_r_func(np.log10(r)) / (r**2 * v)
    # sol = solve_ivp(func, [r[0], r[-1]], [v_crit], method="RK45", vectorized=True, dense_output=True)
    # v_new = sol.sol(r)[0]

    integrand = (Gamma_r - 1) / r ** 2
    integral = np.cumsum((r[1:] - r[:-1]) * (integrand[:-1] + integrand[1:]) / 2)
    integral = np.insert(integral, 0, 0)
    v_squared = v_crit ** 2 + 2 * GM * integral
    v_new = np.sqrt(np.maximum(v_squared, 0))

    vinf_new = v_new[-1] / 1e5
    return v_new, vinf_new


def get_TLucy(teff, r):
    """
    Gives the temperature structure as function of the radius following Lucy(ish)
    """
    return teff * (0.5 * (1 - (1 - r**-2)**0.5))**0.25


def make_result_plot(x, y, xlabel=r"r / $R_\star$", ylabel="", hline=None, hline_label=None, figname="test",
                     final_label=None):
    """
    Makes a simple plot showing the results of the iterations
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    norm = colors.Normalize(vmin=0, vmax=len(y) - 1)
    cmap = cm.get_cmap('viridis', len(y))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, yprof in enumerate(y[:-1]):
        ax.plot(x, yprof, color=cmap(norm(i)), lw=1)

    ax.plot(x, y[-1], color='black', lw=2.5, label=final_label)
    if hline is not None:
        ax.axhline(hline, ls="--", c="k", label=hline_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = plt.colorbar(sm, ax=ax, pad=0.03)
    cbar.set_label("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{figname}.png', dpi=300, bbox_inches="tight")
    plt.close()
    # plt.show()


def refine_vinf(lum, Teff, Mstar, Zstar, Yhe, workdir="tmp_refine", N=20, max_iter=50, tol=1e-3):

    start = time()
    result = run_mcak(lum, Teff, Mstar, Zstar, Yhe, workdir)
    Mdot_time = time()

    if result["fail"]:
        print("Initial MCak run failed:", result["fail_reason"])
        return

    mdot = result["mdot"] * cgs.Msun / cgs.year  # to cgs
    v_inf = result["vinf"] * 1e5  # converting to cgs
    v_crit = result["v_crit"] * 1e5  # converting to cgs
    kap_e = result["kappa_e"]
    rho = result["density"]
    R_star = result["R_star"] * cgs.Rsun
    Qbar = result["Qbar"]
    alpha = result["alpha"]
    Q0 = result["Q0"]
    gamma_e = result["Gamma_e"]

    print("Initial rho:", rho)
    print("Initial T_crit", result["t_crit"])
    v_inf_orig = np.copy(v_inf)
    # print(Qbar, gamma_e, v_inf, v_crit, mdot, R_star)

    r = np.geomspace(R_star, 10 * R_star, N)
    # T_r = get_TLucy(Teff, r / R_star)
    T_r = Teff * np.ones(N)

    # Default start values
    M_t = np.zeros(N)
    kap_e = kap_e * np.ones(N)

    v_prev = None
    vinf_list = []
    M_t_final = None

    v_profiles = []
    density_profiles = []
    Mt_values = []
    t_values = []
    kap_e_profiles = []

    for j in range(max_iter):
        if j == 0:
            v = v_crit + (v_inf - v_crit) * (r - R_star) / (r[-1] - R_star)
        else:
            v = v_new

        rho_r = mdot / (4 * np.pi * r ** 2 * v)

        dv_dr = np.gradient(v, r)

        t = kap_e * rho_r * cgs.c / dv_dr
        lgt = np.log10(np.clip(t, 1e-10, None))
        for i, (T_i, rho_i, t_i) in enumerate(zip(T_r, rho_r, lgt)):
            parameters = get_Mforce_parameters(T_i, rho_i, t_i, workdir)
            logt, Mt_i, kap_e_i = run_mforce(parameters)
            kap_e[i] = kap_e_i[0]
            M_t[i] = Mt_i[0]

        v_new, v_inf_new = mom_eq(r, M_t, v_crit, Mstar, Qbar, gamma_e)

        vinf_list.append(v_inf_new)

        if v_prev is not None:
            dv_inf_rel = np.abs((v_inf_new * 1e5 - v_inf) / v_inf)
            print(f"Iter {j}: v_inf = {v_inf_new:.1f} km/s, Δv_inf = {dv_inf_rel:.2e}")
            if dv_inf_rel < tol:
                break

        v_inf = v_inf_new * 1e5
        v_prev = v
        M_t_final = M_t
        v_profiles.append(v_new.copy())
        density_profiles.append(rho_r.copy())
        Mt_values.append(M_t.copy())
        t_values.append(lgt.copy())
        kap_e_profiles.append(kap_e.copy())

    vinf_time = time()

    print(f"Mass-loss rate determination took {Mdot_time - start:.3g} seconds")
    print(f"Terminal velocity determination took {vinf_time - Mdot_time:.3g} seconds")
    print(f"Total time: {vinf_time - start:.3g}")

    # Plot results
    # fig, axs = plt.subplots(1, 1, figsize=(12, 5))
    # fig, axs = plt.subplots(figsize=(7,5))

    # axs.plot(r / R_star, v_new / 1e5, color='blue')
    # axs.set_xlabel(r"r / $R_\star$")
    # axs.set_ylabel(r"$v\ [\mathrm{km/s}]$")
    # plt.savefig('terminal.png')
    # axs.legend()

    # axs[1].plot(np.log10(t), np.log10(M_t_final), color='darkgreen', label="$M(t)$")
    # axs[1].set_xlabel(r"$\log_{10}(t)$")
    # axs[1].set_ylabel(r"$\log_{10}(M(t))$")
    # axs[1].set_title("Force multiplier")
    # axs[1].legend()

    # plt.tight_layout()

    make_result_plot(r / R_star, np.array(v_profiles) * 1e-5,
                     ylabel=r"$v\ [\mathrm{km/s}]$",
                     hline=v_inf_orig * 1e-5, hline_label="Initial $v_\infty$",
                     figname="Terminal_velocity")

    make_result_plot(r / R_star, np.log10(density_profiles),
                     ylabel=r"$\log \rho$",
                     hline=np.log10(rho), hline_label="Base density",
                     figname="Density")

    make_result_plot(r / R_star, np.log10(Mt_values),
                     ylabel=r"$\log M_t$",
                     figname="Mt_convergence")

    make_result_plot(r / R_star, t_values,
                     ylabel=r"$\log t$",
                     hline=np.log10(result["t_crit"]), hline_label="t_crit",
                     figname="t_convergence")

    make_result_plot(r / R_star, kap_e_profiles,
                     ylabel=r"$\kappa_e$",
                     figname="kap_e")

    return r / R_star, v_new / 1e5, M_t_final, np.log10(t), vinf_list[-1]


if __name__ == "__main__":
    lum = 1e6
    Teff = 35000
    Mstar = 50
    Zstar = 1.
    Yhe = 0.1

    refine_vinf(lum, Teff, Mstar, Zstar, Yhe)
