import numpy as np
import subprocess
import matplotlib
matplotlib.use('Agg') # Use this backend, as it is not interactive.
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
import cgs_constants as cgs
from scipy.optimize import curve_fit
plt.rcParams.update({ 'axes.linewidth':1.2, 'xtick.direction': 'in', 'ytick.direction': 'in','xtick.top': True, 'ytick.right': True, 'xtick.minor.visible':True, 'ytick.minor.visible':True, 'xtick.major.size' : 6, 'xtick.major.width' : 1, 'ytick.major.size' : 8, 'ytick.major.width' : 1, 'xtick.minor.size' : 3.5, 'xtick.minor.width' : 0.6, 'ytick.minor.size' : 3.5, 'ytick.minor.width' : 0.6})
plt.rcParams.update({'font.size': 15})
import os
from mforce import get_force_multiplier
import gc


# Atomic masses of elements (in atomic mass units, amu)
atomic_masses = {'H': 1.008,   'HE': 4.003, 'LI': 6.941,  'BE': 9.012,  'B':  10.811, 'C':  12.011, 'N': 14.007,
                 'O': 16.000,  'F': 18.998, 'NE': 20.180, 'NA': 22.990, 'MG': 24.305, 'AL': 26.982, 'SI': 28.085,
                 'P': 30.974,  'S': 32.066, 'CL': 35.453, 'AR': 39.948, 'K': 39.098,  'CA': 40.078, 'SC': 44.956,
                 'TI': 47.880, 'V': 50.941, 'CR': 51.996, 'MN': 54.938, 'FE': 55.847, 'CO': 58.933, 'NI': 58.690,
                 'CU': 63.546, 'ZN': 65.390}

# This is a dictionary with dummy results in case things crash, so there is something to work with
DUMMY_RESULTS = {"Iteration": -1,
                 "rho": np.nan,
                 "gamma_e*(1+qbar)": np.nan,
                 "rel_mdot": np.nan,
                 "rel_rho": np.nan,
                 "kappa_e": np.nan,
                 "Gamma_e": np.nan,
                 "vesc": np.nan,
                 "rat": np.nan,
                 "phi_cook": np.nan,
                 "R_star": np.nan,
                 "log_g": np.nan,
                 "Qbar": np.nan,
                 "alpha": np.nan,
                 "Q0": np.nan,
                 "vinf": np.nan,
                 "t_crit": np.nan,
                 "v_crit": np.nan,
                 "density": np.nan,
                 "mdot": np.nan,
                 "Zmass": np.nan,
                 "Zscale": np.nan,
                 "alphag": np.nan,
                 "alpha2": np.nan,
                 "warning": False,
                 "fail": True,
                 "fail_reason": ""}


class color:
   # Small class to help emphasize certain printed values
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def write_input_file(filename, params):
    """Write input parameters for MForce to the specified file."""
    with open(filename, 'w') as f:
        f.write("&init_param\n")
        for key, value in params.items():
            f.write(f"{key} = {value}\n")
        f.write("/\n")


def lgM(lgt, alpha, Q0):
    """Compute the logarithmic form of the line force multiplier M(t)."""
    t = 10.0**lgt
    lgM = np.log10(1 / (1 - alpha)) + np.log10((1 + Q0 * t)**(1 - alpha) - 1) - np.log10(Q0 * t)
    return np.nan_to_num(lgM, nan=0.0)


def fit_data(file_path, t_cri):
    """    
    Fit the data from the output file and extract line-force parameters Qb, alpha, and Q0.
    Adjust the fitting range based on t_cri if provided; otherwise, default to 1e-3 Qb.
    """
    # Load in data
    lgt, Mt = np.loadtxt(file_path, unpack=True)
    Qb = np.max(Mt)
    min_mt = Qb / 10**3
    if t_cri is None:
        indices = np.where(Mt >= min_mt)[0]
    else:
        # When fitting based on the t_cri, fit until factor * t_cri to make sure the critical point is covered
        factor = 3.
        lg_tcri = np.log10(t_cri * factor)
        indices = np.where(lgt <= lg_tcri)[0]
        if len(indices) > 0:
            indices = np.arange(0, indices[-1] + 3)

        indices2 = np.where(Mt >= min_mt*100.)[0]
        if len(indices2) > len(indices):
            indices = indices2
    if len(indices) == 0:
        raise ValueError("No data points remain after applying t_cri and min_mt filters")

    lgt_filtered = lgt[indices]
    Mt_filtered = Mt[indices]
    fit_max = max(lgt_filtered)
    lgMt_filtered = np.log10(Mt_filtered / Qb)
    p0 = (0.67, 200)
    # Limits on alpha and Q0
    bounds = ([0.01, 1e-5], [0.99, 1e8])
    
    try:
        popt, _ = curve_fit(lgM, lgt_filtered, lgMt_filtered, p0=p0, bounds=bounds, method="trf")
    except RuntimeError:
        # JS-comment: this needs to be looked at, since when resorting to this method it’s typically
        # for alpha --> 1, and finding fit alpha=1 then crashes code since that’s a diverging
        # limit in the theory. ugly hack for now.
        # but generally: need bounds here as well
        popt, _ = curve_fit(lgM, lgt_filtered, lgMt_filtered, p0=p0, method="lm")

    alpha, Q0 = popt
    alpha_g = alpha
    # Compute alternative alpha_2 as local finite-difference slope around t_cri.
    # We use the two points in lgt_filtered whose values are closest to log10(t_cri)
    target = np.log10(t_cri)
    sorted_idx = np.argsort(np.abs(lgt_filtered - target))
    if len(sorted_idx) < 2:
        alpha_2 = np.nan
        print("Not enough data points to compute local slope alpha_2.")
    else:
        i1, i2 = sorted_idx[:2]
        # Ensure that the points are in increasing order of lgt
        if lgt_filtered[i1] > lgt_filtered[i2]:
            i1, i2 = i2, i1
        # Compute the finite difference slope
        delta = lgt_filtered[i2] - lgt_filtered[i1]
        alpha_2 = (lgMt_filtered[i1] - lgMt_filtered[i2]) / delta

    if alpha > 0.985:
        if alpha_2 <= 0.985:
            print("global alpha too close to diverging limit, resorting"
                "to local alpha at t_cri:", alpha_2)
            alpha = alpha_2
        else:
            alpha = 0.99
            print("too close to alpha divergence limit, setting =",alpha)
    print("final alpha, alpha-local:", alpha, alpha_2)
    return Qb, alpha, Q0, lgt_filtered, alpha_g, alpha_2


def plot_fit(file_path, alpha, Q0, Qb, iteration, t_cri, random_subdir, lgt_filtered):
    """
    Plot original data and reconstructed curve for visual comparison.
    """
    # Load original data
    lgt, M_original = np.loadtxt(file_path, unpack=True)

    M_reconstructed = Qb * 10**lgM(lgt_filtered, alpha, Q0)

    plt.figure(file_path, figsize=(8, 6))
    plt.semilogy(lgt, M_original, marker='H', color='g', markerfacecolor='yellowgreen',
                 markeredgecolor='darkolivegreen', label=r'$M(t)$', markersize=10)
    plt.semilogy(lgt_filtered, M_reconstructed, color='k', label=r'$M_{\rm FIT}(t)$', linewidth=2)

    # Add vertical line for t_cri
    if t_cri is not None:
        plt.axvline(x=np.log10(t_cri), color='#892bed', linestyle='--', linewidth=2,label = fr"$\log_{{10}} t_{{\rm cri}}$: {np.around(np.log10(t_cri),2)}")
        
    # Add horizontal dashed line at max M_reconstructed
    max_M_reconstructed = np.max(M_reconstructed)
    plt.axhline(y=max_M_reconstructed, color='#f08205', linestyle='-.', linewidth=2, label = fr"$\bar{{Q}}$: {np.around(Qb,2)}")

    plt.xlabel(r"$ \log_{10} (t = \frac{\kappa_e \rho c}{dv/dr})$", fontsize=18)
    plt.ylabel(r"$ \log_{10} M(t)$", fontsize=18)
    plt.legend(fontsize=16, ncol=2)
    plt.tight_layout()

    plt.savefig(f'{random_subdir}/Mt_fit_{iteration}.png')
    plt.close()


def plot_convergence(random_subdir, it_num, mdot_num, qbar_num, alpha_num, q0_num, eps_num, delrho_num):
    """
    Plots the convergence of the various relevant parameters.
    """
    fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True, num=random_subdir)
    # Flatten axes for easy indexing
    axes = axes.flatten()
    axes[0].plot(it_num, np.log10([mdot * cgs.year / cgs.Msun for mdot in mdot_num]), marker="D", markersize=10,
                 markerfacecolor='gray', markeredgecolor='k', color='k', linestyle='--', linewidth=2)
    axes[1].plot(it_num, np.log10(qbar_num), marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k',
                 linestyle='--', linewidth=2)
    axes[2].plot(it_num, alpha_num, marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k',
                 linestyle='--', linewidth=2)
    axes[3].plot(it_num, np.log10(q0_num), marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k',
                 linestyle='--', linewidth=2)
    axes[4].plot(it_num, np.log10(eps_num), marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k',
                 color='k', linestyle='--', linewidth=2)
    axes[5].plot(it_num, np.log10(delrho_num), marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k',
                 color='k', linestyle='--', linewidth=2)

    axes[0].set_ylabel(r"$\dot{M} [M_\odot/yr]$")
    axes[1].set_ylabel(r"$\log_{10}(\bar{Q})$")
    axes[2].set_ylabel(r"$\alpha$")
    axes[3].set_ylabel(r"$\log_{10}(Q_0)$")
    axes[4].set_ylabel(r"log $\epsilon_{\dot{M}}$")
    axes[5].set_ylabel(r"log $\epsilon_{\rho}$")
    # Set x-label for the bottom row only
    for ax in axes[-2:]:
        ax.set_xlabel("Iteration")
    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(useOffset=False, style='plain', axis='both')

    plt.tight_layout()
    plt.savefig(f"{random_subdir}/sim_log.png")
    plt.close()


def cak_massloss(lum, qbar, q0, alpha, gamma_e, rat):
    """Calculate mass-loss rate using the CAK formalism. [CITE]"""
    alp = alpha / (1 - alpha) 
    ge = gamma_e / (1 - gamma_e)
    alm = (1 / alpha) - 1
    
    mcak = (lum / cgs.c**2) * alp * ((qbar * ge)**alm)
    fd = 1 / (1 + alpha)
    
    mfd_cak = (fd**(1 / alpha)) * mcak
    mfd_cak *= (1 + 4 * np.sqrt(1 - alpha) / alpha * rat)
    cut = qbar / q0
    mfd_cak *= cut 
    return mfd_cak, cut


def construct_output_filename(T, D):
    """Generate output filename from temperature and density."""
    return f"Mt_{T:.2f}_{D:.1f}"


def radius_calc(lum, teff):
    """Determine the radius based on luminosity and temperature."""
    radius = np.sqrt(lum / (4 * np.pi * cgs.sb * teff**4))
    return radius


def vinf_Kudritzki(alpha, vesc):
    """Determines the terminal wind speed based on the alpha and escape velocity following Kudritzki (YEAR)"""
    f1 = 1.6 * (1 - 0.75 * alpha)
    f3 = 1 - 0.3 * np.exp(-vesc / 300 / 1e5)
    vinf = 2.25*alpha/(1-alpha)*vesc*f1*f3
    return vinf


def read_kappa(file_path):
    """Reads the kappa_e value from the MForce output file"""
    data = np.loadtxt(file_path + '/' + 'Ke_TD', unpack=True)
    kappa_e = data[1,1]
    return kappa_e


def run_mforce(parameters):
    """
    Translates the dictionary of parameters to the input of the get_force_multiplier function and calls the function.
    Note that the order can be confusing, do not assume what it would be, check it if you feel the need to change things
    """
    get_force_multiplier(parameters["lgTmin"],
                         parameters["lgTmax"],
                         parameters["lgDmin"],
                         parameters["lgDmax"],
                         parameters["lgttmin"],
                         parameters["lgttmax"],
                         parameters["Ke_norm"],
                         parameters["X_mass"],
                         parameters["Z_mass"],
                         parameters["N_tt"],
                         parameters["N_lgT"],
                         parameters["N_lgD"],
                         parameters["ver"],
                         parameters["DIR"])


def main(lum, T_eff, M_star, Z_star, Z_scale, Yhe, random_subdir, does_plot, max_iterations=15, tolerance=1e-3, logger=None):
    """
    Main function that calculates the mass-loss rate based on the Luminosity, Effective temperature, Mass, and
    Metallicity. Calls the MForce code and iterates to converge to a consistent mass-loss rate. Creates a temporary
    directory to put output files in.
    NOTE: The actual abundances used are taken from a separate input, which gets written to a temp file.
    """
    if logger is not None:
        log = True
    else:
        log = False

    # Making a temporary directory
    os.makedirs(random_subdir, exist_ok=True)
    
    # Constant: scaling metallicity to solar
    Z_asplund = 0.01334462136084096e0

    # Some default values to be returned in the end
    warning = False
    failure_reason = ""
    fail = True

    if log:
        logger.debug(f"{color.GREEN}{color.BOLD}"
                     f"Running simulation with Luminosity={lum}, T_eff={T_eff}, M_star={M_star}, Yhe={Yhe}"
                     f"{color.END}")

    # Convert some units of input if necessary
    lum = lum * cgs.Lsun
    M_star = M_star * cgs.Msun
    Z_scale = Z_scale * Z_asplund  # This is the scaled metallicity - in mass fraction
    R_star = radius_calc(lum,T_eff)
    # Estimate the Helium ionization stage for the particle and electron density
    Ihe = 2
    if T_eff < 2.5e4:
      Ihe = 1
    mu = (1. + 4. * Yhe) / (2. + Yhe * (1. + Ihe))

    # Initial guess mass loss rate
    mdot_initial = 4. * np.pi * R_star**2. * 100. * 1.e7
    # Initial estimate of kappa_e and Gamma factor
    kap_e = cgs.sigth / cgs.mass_p * (1. + Ihe * Yhe) / (1. + 4. * Yhe)
    gamma_e = kap_e * lum/(4. * np.pi * cgs.G * M_star * cgs.c)

    # Make sure we always iterate at least once to get the real gamma_e
    if gamma_e > 0.95:
        gamma_e = 0.95

    if log:
        logger.debug(f"Initial Gamma_e estimate: {gamma_e}, note: this is capped at 0.95")

    cgas = np.sqrt(cgs.kb * T_eff / (mu * cgs.mass_p))
    v_esc = np.sqrt(2.0 * cgs.G * M_star / R_star * (1.0 - gamma_e))
    rat = v_esc / cgas

    # Initial guesses of line force parameters and mass-loss rate
    qbar = (Z_star / Z_asplund) * 1000. + 1e-8
    alpha = 2./3.
    q0 = qbar
    # ----------
    if log:
        logger.debug("Estimating CAK massloss")
    mdot, cut = cak_massloss(lum, qbar, q0, alpha, gamma_e, 1. / rat)
    if log:
        logger.debug(f"Estimate: {mdot}, {cut}")
    phi_cook = 3.0 * rat**(0.3 * (0.36 + np.log10(rat)))
    v_cri = (phi_cook / (1 - alpha))**0.5
    rho_initial = mdot / (4 * np.pi * v_cri * cgas * R_star**2)
    t_cri = kap_e * cgs.c * rho_initial / (v_cri * cgas / R_star)

    if log:
        logger.debug(f'rho_initial: {rho_initial}, t_cri: {t_cri}')

    lgTeff = np.log10(T_eff)
    lgrho_ini = np.log10(rho_initial)

    # Create dictionary with input parameters for MForce.
    parameters = {"lgTmin": f"{lgTeff:.3E}",
                   "lgTmax": f"{lgTeff:.3E}",
                   "N_lgT": "1",
                   "lgDmin": f"{lgrho_ini:.5E}",
                   "lgDmax": f"{lgrho_ini:.5E}",
                   "N_lgD": "1",
                   "lgttmin": "-8.0",
                   "lgttmax": "10",
                   "N_tt": "50",
                   "Ke_norm": "-10",
                   "X_mass": "0.7",
                   "Z_mass": f"{Z_star:.5E}",
                   "ver": False,
                   "DIR": f"{random_subdir}/output"}

    input_file = os.path.join(random_subdir, "in")

    # Initializing some density and mass-loss rate
    rho = rho_initial
    mdot = mdot_initial

    # Initialize some arrays to keep track of during iterations
    it_num = []
    mdot_num = []
    qbar_num = []
    alpha_num = []
    q0_num = []
    eps_num = []
    delrho_num = []

    # Start the interation loop
    for iteration in range(max_iterations):
        # Update to the density to the new values
        parameters.update({"lgDmin": f"{np.log10(rho):.5e}",
                           "lgDmax": f"{np.log10(rho):.5e}"})

        T = float(parameters["lgTmin"])  # lgTmin (assuming lgTmin == lgTmax)
        D = float(parameters["lgDmin"])  # lgDmin (assuming lgDmin == lgDmax)

        output_file = construct_output_filename(T, D)
        directory = parameters["DIR"].strip("'")
        file_path = directory + '/' + output_file
        if log:
            logger.debug(f"Constructed file path: {file_path}")
            logger.debug("Running Mforce!")

        # Write updated parameters and (re)run MForce
        write_input_file(input_file, parameters)
        run_mforce(parameters)

        # kappa_e is read from the data provided by Mforce and the corresponding Gamma_e is updated
        kap_e = read_kappa(parameters["DIR"].strip("'"))
        gamma_e = kap_e * lum / (4. * np.pi * cgs.G * M_star * cgs.c)

        if gamma_e >= 1:
            failure_reason = f" Gamma_e = {np.around(gamma_e, 2)} > 1, not implemented for these regimes"
            if log:
                logger.debug(f"{color.RED}{color.BOLD}Failure: {failure_reason}{color.END}")
            result_dict = dict(DUMMY_RESULTS)
            result_dict["fail"] = True
            result_dict["fail_reason"] = failure_reason
            return f"{failure_reason}", result_dict

        # New escape velocity based on the new gamma is calculated
        v_esc = np.sqrt(2.0 * cgs.G * M_star / R_star * (1.0 - gamma_e))
        rat = v_esc / cgas
        phi_cook = 3.0 * rat**(0.3 * (0.36 + np.log10(rat)))

        # Line force parameters are calculated/fitted
        qbar, alpha, q0, lgt_filtered, alpha_g, alpha_2 = fit_data(file_path, t_cri)

        # New mass-loss rate is calculated the old one stored
        mdot_old = mdot
        mdot, cut = cak_massloss(lum, qbar, q0, alpha, gamma_e, 1./rat)
        mdot = max(mdot, 1.e-16 * cgs.Msun / cgs.year)

        # Calculate what the next density should be based on the current iteration
        v_cri = (phi_cook / (1 - alpha))**0.5
        rho_target = mdot / (4 * np.pi * v_cri * cgas * R_star**2)

        t_cri = kap_e * cgs.c * rho_target / (v_cri * cgas / R_star)

        # Track convergence by checking the change in mass-loss rate and density
        rel_rho = 1. - rho_target / rho
        rel_mdot = 1. - mdot / mdot_old

        # Calculate the terminal velocity
        vinf = vinf_Kudritzki(alpha,v_esc) / 1e5  # in km/s

        if not np.isfinite(rho_target):
            if log:
                logger.error(f"NaN in Rho at iteration: {iteration}")
            result_dict = dict(DUMMY_RESULTS)
            result_dict["fail"] = True
            result_dict["fail_reason"] = f"NaN in Rho at iteration: {iteration}"
            return f"NaN in Rho at iteration: {iteration}", result_dict
            # raise ValueError('NaN in rho')

        # Save data
        it_num.append(iteration)
        mdot_num.append(mdot)
        qbar_num.append(qbar)
        alpha_num.append(alpha)
        q0_num.append(q0)
        eps_num.append(np.abs(rel_mdot))
        delrho_num.append(np.abs(rel_rho))

        # Make some logs
        mdot_lim = gamma_e*(1.+qbar)
        if log:
            logger.debug(f"Iteration =        {iteration}\n"
                         f"rho_target =       {np.log10(rho_target)}\n"
                         f"rho =              {np.log10(rho)}\n"
                         f"gamma_e*(1+qbar) = {mdot_lim}\n"
                         f"rel_mdot =         {rel_mdot}\n"
                         f"rel_rho =          {rel_rho}\n"
                         f"kappa_e =          {kap_e}\n"
                         f"Gamma_e =          {gamma_e}\n"
                         f"vesc =             {v_esc}\n"
                         f"rat =              {rat}\n"
                         f"phi_cook =         {phi_cook}\n"
                         f"R_star =           {R_star/cgs.Rsun}\n"
                         f"log_g =            {np.log10(cgs.G*M_star/R_star**2)}\n"
                         f"Qbar =             {qbar}\n"
                         f"alpha =            {alpha}\n"
                         f"Q0 =               {q0}\n"
                         f"vinf =             {vinf}\n"
                         f"t_crit =           {t_cri}\n"
                         f"density =          {rho}\n"
                         f"Mass loss rate =   {mdot*cgs.year/cgs.Msun}\n"
                         f"Zmass =            {Z_star}\n"
                         f"Zscale =           {Z_scale}")
        if log:
            logger.debug("----------x----------x----------x----------x----------")

        # Make sure at least 3 iterations happen
        if iteration < 3:
            rho = rho_target
            continue

        # Fail reason 1, cannot drive wind
        if iteration >= 3 and mdot_lim < 1:
            fail = True
            failure_reason = "Line-driven mass loss is not possible. Too low luminosity-mass ratio or metallicity."
            break

        # Convergence Criteria
        if iteration >= 3 and abs(rel_rho) <= tolerance and abs(rel_mdot) <= tolerance:
            fail = False
            break

        if iteration == max_iterations - 1 and alpha > 0.985:
            fail = True
            failure_reason = "Alpha parameter-fit too high, approaching theoretical divergence."
            break

        # reduced convergence
        if iteration == max_iterations - 1 and (abs(rel_rho) <= 2.e-1 or abs(rel_mdot) <= 2.e-1):
            fail = False
            warning = True
            if log:
                logger.info(f"{color.YELLOW}{color.BOLD}"
                            f"WARNING: Not converged to required tolerance (1e-3), please inspect final values before use"
                            f"{color.END}")

        if iteration == max_iterations - 1 and (abs(rel_rho) > 2.e-1 and abs(rel_mdot) > 2.e-1):
            fail = False
            failure_reason = ("The model did not converge after the maximum allowed iterations. Use values with care!" 
                              " Consider using expert mode for more insight.")
            break
        
        # Update the density to the target density for the next iteration
        rho = rho_target

    # If no mass-loss rate was calculated, bring the unfortunate news to the people
    if fail and log:
        logger.info(f"{color.RED}{color.BOLD}Failure: {failure_reason}{color.END}")

    # If the calculation was successful make some diagnostic plots and log the parameters values
    else:
        if does_plot:
            plot_convergence(random_subdir, it_num, mdot_num, qbar_num, alpha_num, q0_num, eps_num, delrho_num)
            plot_fit(file_path, alpha, q0, qbar, iteration, t_cri, random_subdir, lgt_filtered)

        if log:
            logger.debug(f"{color.GREEN}{color.BOLD}\n Converged! {color.END}{color.BOLD}{color.BLUE}")
            logger.debug(f'{"Mass-loss rate":>15}{"Qbar":>10}{"alpha":>10}{"Q0":>10}{"vinf":>10}{"zstar":>10}{color.END}'
                         f"{color.BOLD}{color.GREEN}")
            logger.debug(f"{mdot * cgs.year / cgs.Msun:>15.3g}{qbar:>10.3g}{alpha:>10.3g}"
                         f"{q0:>10.3g}{vinf:>10.3g}{Z_star:>10.3g}{color.END}")

    result_dict = {"Iteration": iteration,
                   "rho": np.log10(rho),
                   "gamma_e*(1+qbar)": mdot_lim,
                   "rel_mdot": rel_mdot,
                   "rel_rho": rel_rho,
                   "kappa_e": kap_e,
                   "Gamma_e": gamma_e,
                   "vesc": v_esc / 1.e5,
                   "rat": rat,
                   "phi_cook": phi_cook,
                   "R_star": R_star / cgs.Rsun,
                   "log_g": np.log10(cgs.G * M_star / R_star ** 2),
                   "Qbar": qbar,
                   "alpha": alpha,
                   "Q0": q0,
                   "vinf": vinf,
                   "t_crit": t_cri,
                   "v_crit": v_cri * cgas * 1e-5,
                   "density": rho,
                   "mdot": mdot * cgs.year / cgs.Msun,
                   "Zmass": Z_star,
                   "Zscale": Z_scale,
                   "alphag": alpha_g,
                   "alpha2": alpha_2,
                   "warning": warning,
                   "fail": fail,
                   "fail_reason": failure_reason}

    if warning:
        result_dict["warning_message"] = "WARNING: Not converged to required tolerance (1e-3), please inspect final values before use"
    else:
        result_dict["warning_message"] = ""

    it_num.clear()
    mdot_num.clear()
    qbar_num.clear()
    alpha_num.clear()
    q0_num.clear()
    eps_num.clear()
    delrho_num.clear()
    gc.collect()
    del it_num, mdot_num, qbar_num, alpha_num, q0_num, eps_num, delrho_num
    del rho_initial, mdot, qbar, alpha, q0, parameters
    gc.collect()

    if fail:
        if result_dict["fail_reason"] == "":
            result_dict["fail_reason"] = "Unknown error, check if input values are valid!"
        return f"FAILURE {failure_reason}", result_dict

    return str(random_subdir), result_dict


if __name__ == "__main__":

    lum = float(sys.argv[1])
    T_eff = float(sys.argv[2])
    M_star = float(sys.argv[3])
    Z_star = float(sys.argv[4])
    Z_scale = float(sys.argv[5])
    Yhel = float(sys.argv[6])
    random_subdir = str(sys.argv[7])
    
    main(lum, T_eff, M_star, Z_star, Z_scale, Yhel, random_subdir)