import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from mcak_explore import main as run_mcak, lgM, run_mforce, fit_data, cak_massloss
import cgs_constants as cgs
from scipy.integrate import solve_bvp, cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import newton

import os
os.system("export MFORCE_DIR=MForce-LTE")

default_abundances = {
                "H": 0.7374078505762753, "HE": 0.24924865007787272, "LI": 5.687053212055474e-11, "BE": 1.5816072816463046e-10,
                "B": 3.9638342804111373e-9, "C": 2.3649741118292409e-3, "N": 6.927752331287037e-4, "O": 5.7328054948662952e-3,
                "F": 5.0460905860356957e-7, "NE": 1.2565170515587217e-3, "NA": 2.9227131182144098e-6, "MG": 7.0785262928672096e-4,
                "AL": 5.5631575894102415e-5, "SI": 6.6484690760698845e-4, "P": 5.8243105278933166e-6, "S": 3.0923740022022601e-4,
                "CL": 8.2016309032581489e-6, "AR": 7.3407809644158897e-5, "K": 3.0647973602772301e-6, "CA": 6.4143590291084783e-5,
                "SC": 4.6455339921264288e-8, "TI": 3.1217731998425617e-6, "V": 3.1718648298183506e-7, "CR": 1.6604169480383736e-5,
                "MN": 1.0817329760692272e-5, "FE": 1.2919540666812507e-3, "CO": 4.2131387804051672e-6, "NI": 7.1254342166372973e-5,
                "CU": 7.2000506248032108e-7, "ZN": 1.7368347374506484e-6
            }


def get_Mforce_parameters(temperature, rho, lgt, workdir, Nt=None):
    """"""
    if not isinstance(lgt, (float, int)):
        minlgt = lgt[0]
        maxlgt = lgt[-1]
        if Nt is None:
            Nt = 2
    else:
        minlgt = lgt
        maxlgt = lgt
        Nt = 1
    parameters = {"lgTmin": f"{np.log10(temperature):.3E}",
                  "lgTmax": f"{np.log10(temperature):.3E}",
                  "N_lgT": "1",
                  "lgDmin": f"{np.log10(rho):.5E}",
                  "lgDmax": f"{np.log10(rho):.5E}",
                  "N_lgD": "1",
                  "lgttmin": f"{minlgt:.5E}",
                  "lgttmax": f"{maxlgt:.5E}",
                  "N_tt": f"{Nt}",
                  "Ke_norm": "-10",
                  "ver": False,
                  "DIR": f"{workdir}/output"}
    return parameters


def write_abundance_file(zscale, workdir):
    """Writes the abundance file"""

    abundance_filename = f"{workdir}/output/mass_abundance"

    abundances = {}
    total_metal_mass = 0
    for element, default_value in default_abundances.items():
        if element in ["H", "HE"]:
            abundances[element] = default_value

        else:
            abundances[element] = default_value * zscale
            total_metal_mass += abundances[element]

    abundances["H"] = 1 - abundances["HE"] - total_metal_mass

    with open(abundance_filename, "w") as f:
        for i, (element, value) in enumerate(abundances.items(), start=1):
            f.write(f"{i:2d}  '{element.upper():2s}'   {value:.14f}\n")


class TerminalVelocity:
    """
    A class to calculate an improved terminal wind velocity of an LIME calculation
    """
    def __init__(self, LIME_out, lum, Teff, Mstar, Zstar, Yhe, workdir="tmp_refine",
                 N=15, max_iter=15, tol=1e-3, Ndens=1000, Rout=10, init_beta=0.8,
                 fd_corr=True, use_dvdr_num=True, mode="basic", include_pressure=True,
                 full_new_mdot=False, new_mdot=False, update_v_crit=True):

        self.LIME_out = LIME_out  # run_mcak(lum, Teff, Mstar, Zstar, Yhe, workdir)
        self.Teff = Teff
        self.lum = lum * cgs.Lsun
        self.Mstar = Mstar * cgs.Msun
        self.Npoints = N
        self.max_iter = max_iter
        self.tol = tol
        self.Ndens = Ndens
        self.Rout = Rout
        self.workdir = workdir
        self.fd_corr = fd_corr
        self.use_dv_dr_num = use_dvdr_num
        self.mode = mode
        self.GM = cgs.G * self.Mstar
        self.include_pressure = include_pressure

        # Mass-loss rate update switches
        self.full_new_mdot = full_new_mdot
        self.new_mdot = new_mdot
        self.update_v_crit = update_v_crit

        self.mdot = self.LIME_out["mdot"] * cgs.Msun / cgs.year  # to cgs
        self.mdot_list = [self.LIME_out["mdot"] * cgs.Msun / cgs.year]
        self.v_inf = self.LIME_out["vinf"] * 1e5  # converting to cgs
        self.v_inf_init = self.LIME_out["vinf"]
        self.v_crit = self.LIME_out["v_crit"] * 1e5  # converting to cgs
        self.Qbar = self.LIME_out["Qbar"]
        self.Q0 = self.LIME_out["Q0"]

        self.rho = self.LIME_out["density"]  # Should already be cgs
        self.rho_r = np.zeros(self.Npoints)
        self.fd = np.zeros(self.Npoints)
        self.t_r = np.zeros(self.Npoints)
        self.Rstar = self.LIME_out["R_star"] * cgs.Rsun
        self.alpha = np.ones(self.Npoints) * self.LIME_out["alpha"]
        self.gamma_e = self.LIME_out["Gamma_e"]
        self.t_crit = self.LIME_out["t_crit"]
        self.kap_e = self.LIME_out["kappa_e"] * np.ones(self.Npoints)


        print(f"QBAR: {LIME_out['Qbar']}")
        print(f"Q0: {LIME_out['Q0']}")
        print(f"alpha: {LIME_out['alpha']}")
        print(f"gamma_e: {LIME_out['Gamma_e']}")

        # Get escape and sound speed
        if Teff > 25000:
            Ihe = 2
        else:
            Ihe = 1

        mu = (1. + 4. * Yhe) / (2. + Yhe * (1. + Ihe))
        self.cgas = np.sqrt(cgs.kb * Teff / (mu * cgs.mass_p))
        self.v_esc = np.sqrt(2.0 * cgs.G * self.Mstar / self.Rstar * (1.0 - self.gamma_e))
        surface_gravity = self.GM / self.Rstar**2
        self.scale_height = self.cgas**2 / surface_gravity



        self.tracked_data = {"v":       [],
                             "vdens":   [],
                             "Mt":      [],
                             "Mtdens":  [],
                             "dvdr":    [],
                             "rho":     [],
                             "t":       [],
                             "fd":      [],
                             "Gamma_r": [],
                             "alpha":   [],
                             }


        # self.r0 = self.Rstar / (1 - (self.v_crit / self.v_inf)**(1/init_beta))
        self.r0 = self.Rstar
        print(f"r0: {self.r0 / self.Rstar:.3f} Rstar")
        print(f"r0: {10 * self.scale_height / self.Rstar:.3f} Rstar")
        print(f"logg: {np.log10(surface_gravity):.3f}")
        # Initialize integration grid:
        # self.r = np.geomspace(self.Rstar, self.Rstar * self.Rout, self.Npoints)
        # self.rdens = np.geomspace(self.Rstar, self.Rstar * self.Rout, self.Ndens)
        self.r = self.sample_rgrid(self.r0, self.Npoints)
        self.rdens = self.sample_rgrid(self.r0, self.Ndens)
        self.logr = np.log10(self.r)
        self.logrdens = np.log10(self.rdens)
        self.M_t0 = 10**lgM(np.log10(self.t_crit), self.alpha[0], self.LIME_out["Q0"])
        self.M_t = np.ones(self.Npoints) * self.M_t0
        self.M_tdens = np.ones(self.Ndens) * self.M_t0

        # Get initial values for velocity and velocity gradient,

        # self.v = self.v_crit + (self.v_inf - self.v_crit) * (1 - self.Rstar / self.r)**init_beta
        # self.vdens = self.v_crit + (self.v_inf - self.v_crit) * (1 - self.Rstar / self.rdens)**init_beta

        # self.v = self.v_inf * (1 - self.Rstar / self.r)**init_beta
        # self.vdens = self.v_inf * (1 - self.Rstar / self.rdens)**init_beta

        self.v = self.get_vinit(self.r, init_beta)
        self.vdens = self.get_vinit(self.rdens, init_beta)

        self.init_deep_start(init_beta)
        self.v0 = self.v[0]

        # Initialize dv_dr arrays
        self.dv_dr = np.zeros(self.Npoints)
        self.dv_drdens = np.zeros(self.Ndens)

        # Get the initial estimate for dv/dr from the (numerical) derivative of the beta law.
        self.dv_dr = self.calc_dvdr_numerical()

        # self.dv_dr[0] = self.v_crit / self.Rstar

        # self.dv_drdens = np.gradient(self.vdens, self.rdens)
        # self.dv_dr = np.interp(self.r, self.rdens, self.dv_drdens)
        # self.calc_dvdr()

    def sample_rgrid(self, r0, n):
        """Samples the radial grid points"""
        x = np.linspace(1 - self.Rstar / r0, 1 - 1 / self.Rout, n)
        r = self.Rstar / (1 - x)
        return r

    def get_vinit(self, r, beta):
        """Gives the initial velocity structure following Santolaya-Rey 1997"""
        b = 1 - (self.cgas / self.v_inf)**(1/beta)
        return self.v_inf * (1 - b * self.Rstar / r)**beta

    def get_atmosphere_v(self, r, rho_s):
        """calculates the velocity in the atmosphere"""
        return self.mdot / (4 * np.pi * r**2 * rho_s * np.exp((self.Rstar - r) / self.scale_height))

    def init_deep_start(self, beta):
        """Makes the input in case we start all the way doing in the atmosphere"""

        # Rho at the stellar surface (tau~2/3)
        rho_s = 0.667 / (self.kap_e[0] * self.scale_height)
        print(f"rho_s: {rho_s:.3g}")

        # Get the connection point between atmosphere
        C = self.mdot / (4 * np.pi * rho_s * self.cgas)
        print(C**0.5 / self.Rstar)
        fun = lambda r: r**2 * np.exp((self.Rstar - r) / self.scale_height) - C
        fun_prime = lambda r: 2 * r * np.exp((self.Rstar - r) / self.scale_height) - (r**2 / self.scale_height) * np.exp((self.Rstar - r) / self.scale_height)

        r_sonic = newton(fun, x0=self.Rstar * 1.004, fprime=fun_prime)

        sel = self.rdens < r_sonic
        self.vdens[sel] = self.get_atmosphere_v(self.rdens[sel], rho_s)
        sel = self.r < r_sonic
        self.v[sel] = self.get_atmosphere_v(self.r[sel], rho_s)

        sel = self.rdens >= r_sonic
        self.vdens[sel] = self.get_vinit(self.rdens[sel], beta)
        sel = self.r >= r_sonic
        self.v[sel] = self.get_vinit(self.r[sel], beta)

    def track_data(self):
        """Saves the current values of specified data into lists for later inspection"""
        self.tracked_data["v"].append(np.copy(self.v))
        self.tracked_data["vdens"].append(np.copy(self.vdens))
        self.tracked_data["dvdr"].append(np.copy(self.dv_dr))
        self.tracked_data["Mt"].append(np.copy(self.M_t))
        self.tracked_data["Mtdens"].append(np.copy(self.M_tdens))
        self.tracked_data["t"].append(np.copy(self.t_r))
        self.tracked_data["rho"].append(np.copy(self.rho_r))
        self.tracked_data["fd"].append(np.copy(self.fd))
        self.tracked_data["Gamma_r"].append(np.maximum(self.gamma_e * (1 + self.M_tdens), 1.01))
        self.tracked_data["alpha"].append(np.copy(self.alpha))

    def apply_finite_disk_corr(self):
        """Does the finite disk correction on the force multiplier"""
        sigma = self.dv_dr * self.r / self.v - 1
        mu_sq = 1 - (self.Rstar / self.r)**2
        self.fd = (((1 + sigma)**(1 + self.alpha) - (1 + sigma * mu_sq)**(1 + self.alpha)) /
              (sigma * (1 + self.alpha) * (1 - mu_sq) * (1 + sigma)**self.alpha))
        self.M_t = self.M_t * self.fd

    def apply_numerical_finite_disk_corr(self):
        """Applies an alternative numerical finite disk correction"""
        oma = self.alpha - 1
        opa = self.alpha + 1

        beta_op = (1 - self.v / (self.dv_dr * self.r)) * (self.Rstar / self.r)**2
        sel1 = beta_op > 1.
        sel2 = beta_op < -1e10
        sel3 = beta_op > 1e-3
        sel4 = np.logical_not(sel1) * np.logical_not(sel2) * np.logical_not(sel3)
        self.fd[sel1] = 1. / opa[sel1]
        self.fd[sel2] = ((-beta_op[sel2])**self.alpha[sel2]) / opa[sel2]
        self.fd[sel3] = (1. - (1. - beta_op[sel3])**opa[sel3]) / (beta_op[sel3] * opa[sel3])
        self.fd[sel4] = 1. - 0.5 * self.alpha[sel4] * beta_op[sel4] * (1. + 0.3333333 * oma[sel4] * beta_op[sel4])
        self.M_t = self.M_t * self.fd
        #
        # sigma = self.dv_dr * self.r / self.v - 1
        # mu_sq = 1 - (self.Rstar / self.r)**2
        # fd = (((1 + sigma)**(1 + self.alpha) - (1 + sigma * mu_sq)**(1 + self.alpha)) /
        #       (sigma * (1 + self.alpha) * (1 - mu_sq) * (1 + sigma)**self.alpha))
        #
        # plt.plot(1 - self.Rstar / self.r, self.fd, label="Numerical")
        # plt.plot(1 - self.Rstar / self.r, fd, label="Analytical")
        # plt.legend()
        # plt.show()

    def calc_dvdr(self, r, v, M_t):
        """Calculates the local analytical velocity gradient"""
        Gamma_r = np.maximum(self.gamma_e * (1 + M_t), 1.01)  # Force Gamma_R larger than 1!
        return self.GM * (Gamma_r - 1) / (v * r**2)

    def calc_dvdr_with_gas(self, r, v, M_t, rho):
        """Does the momentum equation including gas pressure"""
        Gamma_r = np.maximum(self.gamma_e * (1 + M_t), 1.01)
        drho_dr = -2 * self.mdot / (4 * np.pi * r**3 * v)  # For now assume v constant?
        pressure_term = drho_dr * self.cgas**2 / (rho * v)
        dvdr = self.GM * (Gamma_r - 1) / (v * r**2) - pressure_term
        return dvdr

    def calc_dvdr_numerical(self):
        """Get the velocity derivative directly from the dens grid"""
        self.dv_drdens = np.gradient(self.vdens, self.rdens)
        return interp1d(self.rdens, self.dv_drdens, kind="linear")(self.r)

    def calc_alpha(self, logt, M_t):
        """Updates the value of alpha based on the slope of M(t)"""
        M_t = np.log10(M_t)
        alpha = (M_t[0] - M_t[-1]) / (logt[-1] - logt[0])
        return alpha

    def solve_iteratively(self):
        """
        Solves the momentum equation iteratively over the whole profile
        """
        for j in range(self.max_iter):
            self.track_data()
            print(self.mdot / cgs.Msun * cgs.year)
            self.rho_r = self.mdot / (4 * np.pi * self.r**2 * self.v)
            # The lower limit of t is fixed to 10**-10
            self.t_r = np.clip(self.kap_e * self.rho_r * cgs.c / self.dv_dr, a_min=1e-10, a_max=None)
            # self.t_r[0] = self.t_crit  # Keep the first point fixed
            for i, (rho_i, t_i) in enumerate(zip(self.rho_r, self.t_r)):
                # if i == 0:
                #     self.M_t[i] = self.M_t0
                # else:
                parameters = get_Mforce_parameters(self.Teff,
                                                   rho_i,
                                                   [np.log10(t_i), np.log10(t_i) + 0.01],
                                                   self.workdir)
                logt, Mt_i, kap_e_i = run_mforce(parameters)
                self.kap_e[i] = kap_e_i[0]
                self.M_t[i] = Mt_i[0]
                self.alpha[i] = self.calc_alpha(logt, Mt_i)

            if self.fd_corr:
                self.apply_finite_disk_corr()
                # self.apply_numerical_finite_disk_corr()

            # Interpolate M_t to the dense grid for integration
            self.interp_Mt()

            # Solve the integral
              # Save the old/initial values
            vdens_new = self.solve_full_momentum_equation()
            v_new = np.interp(self.r, self.rdens, vdens_new)
            v_prev = self.v
            self.vdens = vdens_new
            self.v = v_new
            self.v_inf = self.v[-1]


            if self.use_dv_dr_num:
                self.dv_dr = self.calc_dvdr_numerical()
            else:
                self.dv_dr = self.calc_dvdr(self.r, self.v, self.M_t)
            # self.dv_dr[0] = np.max([self.dv_dr[0], 1.01 * self.v_crit / self.Rstar])

            if self.new_mdot:
                self.get_new_mdot()

            # Check convergence:
            delta_vinf = np.abs(self.v_inf - v_prev[-1]) / self.v_inf
            print(f"Iter {j}: v_inf = {self.v_inf * 1e-5:.1f} km/s, Δv_inf = {delta_vinf:.2e}")
            if delta_vinf < self.tol:
                break

        self.track_data()  # Save the final values

    def interp_Mt(self):
        """Does an interpolation of Mt to the dens grid, does a linear interpolation in log space"""
        interpf = interp1d(self.logr, np.log10(self.M_t), kind="linear")
        self.M_tdens = 10**(interpf(self.logrdens))

    def solve_full_momentum_equation(self):
        """
        Determines the full radial velocity profile based on the current force multipliers
        Does this by calling the chosen solver method using mode.
        """
        if self.mode == "basic":
            return self.trapezoid_solver()

        elif self.mode == "bvp":
            return self.bvp_solver()

        elif self.mode == "rk4":
            if self.include_pressure:
                return self.rk4_solver_with_pressure()
            else:
                return self.rk4_solver()

        else:
            print("This mode is not implemented!")
            return self.vdens

    def trapezoid_solver(self):
        """Does a basic trapezoid integration to find the velocity profile"""
        # solved equation: v**2 / 2 = G * Mstar * integral((gamma_r - 1) / r**2) dr)
        # Due to low dv/dr and absence of gas pressure (?) no wind can be sustained in the inner region,
        # Therefore, add a minimum value of Gamma_r of 1.01.
        Gamma_r = np.maximum(self.gamma_e * (1 + self.M_tdens), 1.01)
        integrand = self.GM * (Gamma_r - 1) / self.rdens ** 2
        new_v = 2 * (cumulative_trapezoid(integrand, self.rdens, initial=0)) ** 0.5 + self.v_crit
        return new_v

    def bvp_solver(self):
        """Uses the scipy solve_bvp function to get the velocity profile."""
        Gamma_r = self.GM * (np.maximum(self.gamma_e * (1 + self.M_tdens), 1.01) - 1)
        Gamma_r_interp = interp1d(self.rdens, Gamma_r, kind="cubic", fill_value="extrapolate")

        fun = lambda r, v: np.vstack(((Gamma_r_interp(r) / (r ** 2 * v[0])),))
        bc = lambda ya, yb: np.array([ya[0] - self.v_crit])

        sol = solve_bvp(fun, bc, self.rdens, np.vstack((self.vdens,)), max_nodes=1000)
        return sol.sol(self.rdens)[0]

    def rk4_solver(self):
        """Does a 4th order Runge Kutta to get the velocity profile"""
        new_v = np.zeros(self.Ndens)
        new_v[0] = self.v_crit
        M_t_interp = interp1d(self.logr, np.log10(self.M_t), kind="linear")

        for i in range(self.Ndens - 1):
            r = self.rdens[i]
            h = self.rdens[i + 1] - r
            v = new_v[i]

            k1 = h * self.calc_dvdr(r, v, 10**(M_t_interp(np.log10(r))))
            k2 = h * self.calc_dvdr(r + 0.5 * h, v + 0.5 * k1, 10**(M_t_interp(np.log10(r + 0.5 * h))))
            k3 = h * self.calc_dvdr(r + 0.5 * h, v + 0.5 * k2, 10**(M_t_interp(np.log10(r + 0.5 * h))))
            k4 = h * self.calc_dvdr(r + h, v + k3, 10**(M_t_interp(np.log10(r + h))))

            new_v[i + 1] = v + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return new_v

    def rk4_solver_with_pressure(self):
        """Does a 4th order Runge-Kutta to get the velocity profile"""
        new_v = np.zeros(self.Ndens)
        new_v[0] = self.v0
        M_t_interp = interp1d(self.logr, np.log10(self.M_t), kind="linear")
        rho_interp = interp1d(self.logr, np.log10(self.rho_r), kind="linear")

        for i in range(self.Ndens - 1):
            r = self.rdens[i]
            h = self.rdens[i + 1] - r
            v = new_v[i]

            k1 = h * self.calc_dvdr_with_gas(r, v,
                                             10**(M_t_interp(np.log10(r))),
                                             10**(rho_interp(np.log10(r))))
            k2 = h * self.calc_dvdr_with_gas(r + 0.5 * h, v + 0.5 * k1,
                                             10**(M_t_interp(np.log10(r + 0.5 * h))),
                                             10**(rho_interp(np.log10(r + 0.5 * h))))
            k3 = h * self.calc_dvdr_with_gas(r + 0.5 * h, v + 0.5 * k2,
                                             10**(M_t_interp(np.log10(r + 0.5 * h))),
                                             10**(rho_interp(np.log10(r + 0.5 * h))))
            k4 = h * self.calc_dvdr_with_gas(r + h, v + k3,
                                             10**(M_t_interp(np.log10(r + h))),
                                             10**(rho_interp(np.log10(r + h))))

            new_v[i + 1] = v + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return new_v

    def get_new_mdot(self):
        """Calculates a new mass-loss rate based on the velocity gradient and t at the inner boundary"""

        rat = self.cgas / self.v_esc
        # Only use the new local alpha value for the new mass loss rate.
        if not self.full_new_mdot:

            self.mdot = cak_massloss(self.lum, self.Qbar, self.Q0, self.alpha[0], self.gamma_e, rat)
            self.mdot_list.append(np.copy(self.mdot))

        else:  # Calculate a full new mass loss rate based on the new density and velocity gradient (so t)

            lgt_range = (-8, 10)
            parameters = get_Mforce_parameters(self.Teff, self.rho_r[0], lgt_range, self.workdir, Nt=50)
            lgt, Mt, kappa_e = run_mforce(parameters)
            Qbar, alpha, Q0, _, _, _ = fit_data((lgt, Mt), self.t_r[0])

            self.mdot = cak_massloss(self.lum, Qbar, Q0, alpha, self.gamma_e, rat)

            print(Qbar, Q0, alpha)

        if self.update_v_crit:
            self.v_crit = self.dv_dr[0] * (self.r0 - self.Rstar)
            self.v = self.v - self.v[0] + self.v_crit

    def plot_results(self, figname="test_figure", show=True):
        """Plots some of the results of the convergence"""
        fig, axarr = plt.subplots(3, 3, figsize=(10,8),
                                  sharex=True, layout="constrained")

        axarr = axarr.ravel()

        self.plot_profile(axarr[0],
                          1 - self.Rstar / self.rdens,
                          np.array(self.tracked_data["vdens"]) * 1e-5,
                          xlabel=None,
                          ylabel=r"$v\ [\mathrm{km/s}]$",
                          hline=self.v_inf_init,
                          hline_label=r"Initial $v_\infty$")

        self.plot_profile(axarr[1],
                          1 - self.Rstar / self.r,
                          self.tracked_data["Mt"],
                          xlabel=None,
                          ylabel=r"$M_t$")

        self.plot_profile(axarr[2],
                          1 - self.Rstar / self.r,
                          np.log10(np.array(self.tracked_data["dvdr"]) * self.Rstar / self.v_inf),
                          ylabel="log dv/dr",
                          xlabel=None)

        axarr[2].axhline(np.log10(self.v_crit / self.Rstar), ls="--", c="k")

        self.plot_profile(axarr[3],
                          1 - self.Rstar / self.r,
                          np.log10(self.tracked_data["t"]),
                          ylabel="log $t$",
                          xlabel=None,
                          hline_label="t_crit",
                          hline=np.log10(self.t_crit))

        self.plot_profile(axarr[4],
                          1 - self.Rstar / self.rdens,
                          self.tracked_data["Gamma_r"],
                          xlabel=None,
                          ylabel="Gamma_r")

        self.plot_profile(axarr[5],
                          1 - self.Rstar / self.r,
                          self.tracked_data["fd"],
                          xlabel=None,
                          ylabel="Finite disk correction")

        self.plot_profile(axarr[6],
                          1 - self.Rstar / self.r,
                          np.log10(self.tracked_data["rho"]),
                          xlabel=r"1 - $R_\star / r$",
                          ylabel=r"log $\rho$")

        sm = self.plot_profile(axarr[7],
                          1 - self.Rstar / self.r,
                          self.tracked_data["alpha"],
                          xlabel=r"1 - $R_\star / r$",
                          ylabel="alpha")

        for r in self.r:
            axarr[0].axvline(1 - self.Rstar / r, ls="--", color="0.5", lw=0.5)

        cbar = fig.colorbar(sm, ax=axarr, shrink=0.5)
        cbar.set_label("Iteration")
        # plt.tight_layout()
        plt.savefig(f"plots/{figname}.pdf", bbox_inches="tight")
        plt.savefig(f"plots/{figname}.png", dpi=150)
        if show:
            plt.show()
        else:
            plt.close()


    def plot_profile(self, ax, x, y, xlabel=r"$r / R_\star$", ylabel="", hline=None, hline_label=None,
                     final_label=None, **kwargs):

        norm = colors.Normalize(vmin=0, vmax=len(y) - 1)
        cmap = plt.get_cmap('viridis', len(y))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)

        for i, yprof in enumerate(y[:-1]):
            ax.plot(x, yprof, color=cmap(norm(i)), lw=1, **kwargs)

        ax.plot(x, y[-1], color='black', lw=2.5, label=final_label, **kwargs)
        if hline is not None:
            ax.axhline(hline, ls="--", c="k", label=hline_label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # cbar = plt.colorbar(sm, ax=ax, pad=0.03)
        # cbar.set_label("Iteration")

        if hline_label is not None and final_label is not None:
            ax.legend()

        return sm


if __name__ == "__main__":

    from time import time

    lum = 10**5.54
    Teff = 43100
    Mstar = 39
    Zstar = 0.2 * 0.014
    Yhe = 0.1
    workdir = "tmp_refine"
    write_abundance_file(0.2, workdir)
    LIME_out = run_mcak(lum, Teff, Mstar, Zstar, Yhe, workdir)

    modes = ("basic", "bvp", "rk4")
    N = 15
    N_vals = [10, 20, 40, 80]
    vinfs = dict(zip(modes, ([], [], [])))
    vinfs["N"] = N_vals
    betas = (0.5, 1, 5)
    for beta in betas:
        for mode in modes:
            start = time()
            print(f"Starting {mode} run with beta={beta}!")
            TWV = TerminalVelocity(LIME_out, lum, Teff, Mstar, Zstar, Yhe, N=N, fd_corr=True, use_dvdr_num=True, mode=mode, init_beta=beta)
            TWV.solve_iteratively()
            TWV.plot_results(show=False, figname=f"test_beta{beta}_{mode}")
            print(f"It took {time() - start:.2f} seconds")
            vinfs[mode].append(TWV.v_inf)

    # import pandas as pd
    #
    # print(vinfs)
    # df = pd.DataFrame(vinfs)
    # print(df)
    # df.to_csv("terminal_velocity_tests.csv")


        # start = time()
        # mode = "basic"
        # print(f"Starting {mode} run!")
        # TWV = TerminalVelocity(LIME_out, lum, Teff, Mstar, Zstar, Yhe, N=N, fd_corr=True, use_dvdr_num=True, mode=mode)
        # TWV.solve_iteratively()
        # TWV.plot_results(show=False, figname=f"test_N{N}_{mode}")
        # print(f"It took {time() - start:.2f} seconds")
        #
        # start = time()
        # mode = "rk4"
        # print(f"Starting {mode} run!")
        # TWV = TerminalVelocity(LIME_out, lum, Teff, Mstar, Zstar, Yhe, N=N, fd_corr=True, use_dvdr_num=True, mode=mode)
        # TWV.solve_iteratively()
        # TWV.plot_results(show=False, figname=f"test_N{N}_{mode}")
        # print(f"It took {time() - start:.2f} seconds")
