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
import tempfile
from mforce import get_force_multiplier

import random
import shutil


# Atomic masses of elements (in atomic mass units, amu)
atomic_masses = { 'H': 1.008,'HE': 4.003, 'LI': 6.941,'BE': 9.012,'B': 10.811,'C': 12.011,'N': 14.007,'O': 16.000,'F': 18.998,'NE': 20.180,'NA': 22.990,'MG': 24.305,'AL': 26.982,'SI': 28.085,'P': 30.974,'S': 32.066,'CL': 35.453,'AR': 39.948,'K': 39.098,'CA': 40.078,'SC': 44.956,'TI': 47.880,'V': 50.941,'CR': 51.996,'MN': 54.938,'FE': 55.847,'CO': 58.933,'NI': 58.690,'CU': 63.546,'ZN': 65.390 }

# Function to read the default abundances
def read_default_abundances(file_path):
    """Read the default metal composition from the file."""
    abundances = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    index, element, abundance = parts
                    abundances[element.strip("'")] = float(abundance)
    except FileNotFoundError:
        print(f"Default abundance file not found at {file_path}.")
        raise
    return abundances


def write_abundances(filename, abundances):
    """
    Write the abundance data to the specified file.

    Parameters:
    - filename: str, the path to the file to write.
    - abundances: dict, a dictionary where keys are element names and values are their abundances.
    """
    with open(filename, 'w') as f:
        for i, (element, abundance) in enumerate(abundances.items(), start=1):
            # Ensure element names are consistently 2 characters wide
            formatted_element = element.ljust(2)
            f.write(f"{i:2d}  '{formatted_element}'   {abundance:.14f}\n")

# Utility function to write parameters to the input file
def write_input_file(filename, params):
    """Write input parameters to the specified file."""
    with open(filename, 'w') as f:
        f.write("&init_param\n")
        for key, value in params.items():
            f.write(f"{key} = {value}\n")
        f.write("/\n")

# Run the Fortran program with the input file
def run_fortran_program(executable, input_file):
    """Run the Fortran program with the specified executable and input file."""
    try:
        result = subprocess.run([executable, "-i", input_file], check=True, text=True, capture_output=True)
        print("Program output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred while running the program:")
        print(e.stderr)

# Define the line force multiplier function
def lgM(lgt, alpha, Q0):
    """Compute the logarithmic form of the line force multiplier M(t)."""
    t = 10.0**lgt
    #print(t,alpha,Q0)
    lgM = np.log10(1/(1-alpha)) + np.log10((1+Q0*t)**(1-alpha) - 1) - np.log10(Q0*t)
    return np.nan_to_num(lgM, nan=0.0)

def fit_data(file_path, t_cri):
    """
    Fit the data from the output file and extract line-force parameters Qb, alpha, and Q0.
    Adjust the fitting range based on t_cri if provided; otherwise, default to 1e-3 Qb.
    """
    data = np.loadtxt(file_path, unpack=True)
    lgt, Mt = data[0], data[1]  
    Qb = np.max(Mt)  
    #a factor of safety for fitting based on t_cri estimate  
    factor = 3.
    
    min_mt = Qb / 10**3
    if t_cri is None:
        indices = np.where(Mt >= min_mt)[0]  
    else:
        #see above 
        lg_tcri = np.log10(t_cri*factor)
        indices = np.where(lgt <= lg_tcri)[0]
        #never fit to less than 0.1*Qbar 
        if len(indices) > 0:
            indices = np.arange(0, indices[-1] + 3)
        #    
        indices2 = np.where(Mt >= min_mt*100.)[0]  
        if len(indices2) > len(indices): 
            indices = indices2

    if len(indices) == 0:
        raise ValueError("No data points remain after applying t_cri and min_mt filters.")
     
    lgt_filtered = lgt[indices]
    Mt_filtered = Mt[indices]
    fit_max = max(lgt_filtered)
    print('lgt Fitted until:',fit_max)

    lgMt_filtered = np.log10(Mt_filtered / Qb)

    p0 = None
    # Limits on alpha and Q0 
    bounds = ([0.01, 1e-5], [0.98, 1e8])

    try:
        popt, _ = curve_fit(lgM, lgt_filtered, lgMt_filtered, p0=p0, bounds=bounds, method='trf')
    except RuntimeError:
        #JS-comment: this needs to be looked at, since when resprting to this method it's typically
        #for alpha --> 1, and finding fit alpha=1 then crashes code since that's a diverging
        #limit in the theory. ugly hack for now. 
        #but generally: need bounds here as well 
        popt, _ = curve_fit(lgM, lgt_filtered, lgMt_filtered, p0=p0, method='lm')
        #popt, _ = curve_fit(lgM, lgt_filtered, lgMt_filtered, p0=p0, bounds=bounds, method='lm')
        
    alpha, Q0 = popt
    if alpha > 0.98:
        alpha = 0.98 
    print('Q0, alpha:',Q0,alpha)
    return Qb, alpha, Q0, lgt_filtered


def plot_fit(file_path, alpha, Q0, Qb, iteration, t_cri, random_subdir):
    """
    Plot original data and reconstructed curve for visual comparison.
    """
    # Load original data
    data = np.loadtxt(file_path, unpack=True)
    lgt, M_original = data[0], data[1]
    
    _, _, _, lgt_filtered = fit_data(file_path, t_cri)
    
    M_reconstructed = Qb * 10**lgM(lgt_filtered, alpha, Q0)
    
    output_file = f"{random_subdir}/M_reconstructed_{iteration}.txt"
    np.savetxt(output_file, np.column_stack((lgt_filtered, M_reconstructed)), 
               header='lgt_filtered\tM_reconstructed', fmt='%.6e', comments='')
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(lgt, M_original, marker='H', color='g', markerfacecolor='yellowgreen',
                 markeredgecolor='darkolivegreen', label=r'$M(t)$', markersize=10)
    plt.semilogy(lgt_filtered, M_reconstructed, color='k', label=r'$M_{\rm FIT}(t)$', linewidth=2)

    # Add vertical line for t_cri
    if t_cri is not None:
        plt.axvline(x=np.log10(t_cri), color='#892bed', linestyle='--', linewidth=2,label = fr"$\log_{{10}} t_{{\rm cri}}$: {np.around(np.log10(t_cri),2)}")
        
    # Add horizontal dashed line at max M_reconstructed
    max_M_reconstructed = np.max(M_reconstructed)
    plt.axhline(y=max_M_reconstructed, color='#f08205', linestyle='-.', linewidth=2, label = fr"$\bar{{Q}}$: {np.around(Qb,2)}")

    
    plt.xlabel(r"$ \log_{10} (t)$", fontsize=18)
    plt.ylabel(r"$ \log_{10} M(t)$", fontsize=18)
    plt.legend(fontsize=16, ncol=2)
    plt.tight_layout()

    plt.savefig(f'{random_subdir}/Mt_fit_{iteration}.png')
    plt.close()

def cak_massloss(lum, qbar, q0, alpha, gamma_e, rat): 
    """Calculate mass-loss rate using the CAK formalism."""
    alp = alpha / (1 - alpha) 
    ge = gamma_e / (1 - gamma_e)
    alm = (1 / alpha) - 1
    
    mcak = (lum / cgs.c**2) * alp * ((qbar * ge)**alm)
    fd = 1 / (1 + alpha)
    
    mfd_cak = (fd**(1/alpha)) * mcak
    mfd_cak *= (1 + 4 * np.sqrt(1 - alpha) / alpha * rat)
    cut = qbar / q0
    mfd_cak *= cut 
    return mfd_cak, cut
    
def construct_output_filename(T, D):
    """Generate output filename from temperature and density."""
    return f"Mt_{T:.2f}_{D:.1f}"

def radius_calc(lum,teff):
  radius = np.sqrt(lum/(4*np.pi*cgs.sb*teff**4))
  return radius

def Vink(teff,lum,mstar,game,z):
  "Vink et al 2001"
  
  z = z/ 0.0134 # the scaling is done based on the rates of Asplund et al 2009
  lum/=cgs.Lsun
  mstar/=cgs.Msun
  #JS-note: this needs to be better implemented, see his paper
  #(the range 22500 is not fixed..)
  #JS-ADD: Dependence of limit (Vink+ 2001):
  #rhom = -13.636 + 0.889*np.log10(z)
  rhom = -14.94 + 0.85*np.log10(z) + 3.2*game 
  teff_lim = (61.2 + 2.59*rhom)*1000.
    #JS-NOTE: TYPOS IN VINF OVER VESC RATIOS ABOVE? REMEMBER TO ASK OLIVIER!
    #--------
    #--------
  if teff > teff_lim:
  #if teff >27500:
      logmdot = -6.697+ 2.194*np.log10(lum/10**5)-1.313*np.log10(mstar/30)-1.226*np.log10(2.6/2.0)+0.933*np.log10(teff/40000)-10.92*np.log10(teff/40000)**2+0.85*np.log10(z)
  else:
      logmdot = -6.688+ 2.210*np.log10(lum/10**5)-1.339*np.log10(mstar/30)-1.601*np.log10(1.3/2.0)+1.07*np.log10(teff/20000)+0.85*np.log10(z)
        
  return 10**logmdot

# Bjorklund et al 2020
def bjorklund(lum,mstar,teff,gamma_e,z):
  z = z/ 0.0134 # the scaling is done based on the rates of Asplund et al 2009
  lum = lum/cgs.Lsun
  z_sol = 1.e0 # solar metallicity scaled to 1
  mdot = - 5.52 + 2.39*np.log10(lum/(1e6)) - 1.48*np.log10((mstar/cgs.Msun)*(1-gamma_e)/45.) + 2.12*np.log10(teff/4.5e4) + (0.75-1.87*np.log10(teff/4.5e4))*(np.log10(z/z_sol))  
  return 10**mdot
  
# kriticka et al 2019
def Kriticka(lum, teff, z):
    lum = lum/cgs.Lsun
    z = z/0.0134
    logmdot = -13.82+0.358*np.log10(z)+(1.52-0.11*np.log10(z))*np.log10(lum/10**6)+13.82*np.log10((1+0.73*np.log10(z))*np.exp(-(teff-14160)**2/3580**2)+3.84*np.exp(-(teff-37900)**2/(56500**2)))
    return 10**logmdot

def calculate_metallicity(number_abundances, atomic_masses):
    """
    Calculate metallicity by summing mass abundances of the metals.

    Args:
        number_abundances (dict): Dictionary of number abundances.
        atomic_masses (dict): Dictionary of atomic masses.

    Returns:
        float: Metallicity (Z) value.
    """
    metals = {
        'LI': 6.941,'BE': 9.012,'B': 10.811,'C': 12.011,'N': 14.007,'O': 16.000,'F': 18.998,'NE': 20.180,'NA': 22.990,'MG': 24.305,'AL': 26.982,'SI': 28.085,'P': 30.974,'S': 32.066,'CL': 35.453,'AR': 39.948,'K': 39.098,'CA': 40.078,'SC': 44.956,'TI': 47.880,'V': 50.941,'CR': 51.996,'MN': 54.938,'FE': 55.847,'CO': 58.933,'NI': 58.690,'CU': 63.546,'ZN': 65.390 }
    
    total_mass_abundance = sum(
        number_abundances[element] * atomic_masses[element]
        for element in number_abundances if element in atomic_masses
    )

    metallicity = sum(
        number_abundances[element] * atomic_masses[element] / total_mass_abundance
        for element in metals if element in number_abundances
    )

    return metallicity
  

# Function to let user provide custom abundances
def get_user_abundances(default_abundances):
    """Prompt the user to input custom metal abundances."""
    user_abundances = default_abundances.copy()
    print("Provide new abundances for elements or press Enter to keep default values:")
    for element, default_value in default_abundances.items():
        user_input = input(f"{element} (default {default_value}): ")
        if user_input.strip():
            try:
                user_abundances[element] = float(user_input)
            except ValueError:
                print(f"Invalid input for {element}. Keeping default value {default_value}.")
    return user_abundances

def read_kappa(file_path):
  data = np.loadtxt(file_path + '/' + 'Ke_TD' , unpack=True)
  kappa_e = data[1,1]
  return kappa_e


def run_mforce(parameters):
    """
    Translates the dictionary of parameters to the input of the get_force_multiplier function and calls the function.
    Note that the order can be confusing, do not assume what it would be, check it if you feel the need to change things
    """

    print("THE RANDOM DIRECTORY IS: ", parameters["DIR"])

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


# Integrating the functionality into the main workflow

def main(lum, T_eff, M_star, Z_star, Z_scale, Yhel, random_subdir):

    # Making a temporary file
    os.makedirs(random_subdir, exist_ok=True) 

    log_file = os.path.join(random_subdir, "simlog.txt") 
    
    # scaling metallicity to solar
    Z_asplund = 0.01334462136084096e0 
       
    with open(log_file, 'w') as log: 
        def log_print(*args, **kwargs):
            """Print and log to file simultaneously."""
            print(*args, **kwargs)
            print(*args, **kwargs, file=log)
            
        log_print(f"Running simulation with Luminosity={lum}, T_eff={T_eff}, M_star={M_star}, Yhe={Yhel}")    
    
        lum = lum*cgs.Lsun 
        T_eff = T_eff
        M_star = M_star* cgs.Msun
        
        "This is the actual metallicity - in mass fraction"
        Z_star = Z_star 
        
        "This is the scaled metallicty - in mass fraction"  
        Z_scale = Z_scale * Z_asplund
    
        max_iterations = 25 #1000
        tolerance = 1.e-3
    
        R_star = radius_calc(lum,T_eff)
        
        "Helium abundances are read from the user"
        
        Yhe = Yhel 
        Ihe = 2
        if T_eff < 2.5e4:
          Ihe = 1
        mu = (1.+4.*Yhe)/(2.+Yhe*(1.+Ihe))
        
         
        "Initial density estimate from Lattimer and Cranmer, but scaled to metallicity"         
        rho_initial = (Z_star/Z_asplund) * 6.33e-16*(T_eff/10000.)**6.2
        #JS-Dec16, Z-scaling needs to be normalised        
        mdot_initial = 4.*np.pi*R_star**2.*100.*1.e5
        #JS Dec: better, use initial line-force parameters:
        kap_e = cgs.sigth/cgs.mass_p*(1.+Ihe*Yhe)/(1.+4.*Yhe)
        gamma_e = kap_e*lum/(4.*np.pi*cgs.G*M_star*cgs.c)
        if (gamma_e >= 1):
            log_print('gamma_e > 1', gamma_e, kap_e)
            sys.exit(0)         
        cgas = np.sqrt(cgs.kb * T_eff / (mu * cgs.mass_p))
        v_esc = np.sqrt(2.0 * cgs.G * M_star / R_star * (1.0 - gamma_e))
        rat = v_esc / cgas
        #initial guesses 
        qbar = (Z_star/Z_asplund)*1000.
        alpha = 2./3.
        q0 = qbar 
        # ----------                
        mdot, cut = cak_massloss(lum, qbar, q0, alpha, gamma_e, 1./rat)
        phi_cook = 3.0 * rat**(0.3 * (0.36 + np.log10(rat)))
        v_cri = (phi_cook/(1-alpha))**0.5
        log_print('rho_initial LC:',rho_initial)
        rho_initial = mdot / (4 * np.pi * v_cri * cgas * R_star**2)
        t_cri = kap_e*cgs.c*rho_initial/(v_cri*cgas/R_star)
        log_print('rho_initial, t_cri:',rho_initial, t_cri)
        
        lgTeff = np.log10(T_eff)
        lgrho_ini = np.log10(rho_initial)


        parameters = {
           "DIR": f"{random_subdir}/output",
           "lgTmin": f"{lgTeff:.3E}",
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
           "ver": ".FALSE."
           }
        input_file = os.path.join(random_subdir, "in")
        
            
        #log_print(f"Gamma={gamma_e}, kappa={kap_e}, mu={mu}, cgas={cgas}, vesc={v_esc}")
    
        iteration = 0
        eps = 1.e-3
        np_mean = 50     
        
        "Initializing some density and massloss"
        rho = rho_initial 
        mdot = mdot_initial
        
        it_num = []
        mdot_num=[]
        qbar_num = []
        alpha_num = []
        q0_num = []
        #JS: Dec 16 
        eps_num = []
        delrho_num = [] 

        #Using start guess also for t_cri 
        #t_cri = None
        for iteration in range(max_iterations):
    
            parameters.update({
              "lgDmin": f"{np.log10(rho):.5e}",
              "lgDmax": f"{np.log10(rho):.5e}",
             })
        
            T = float(parameters["lgTmin"])  # lgTmin (assuming lgTmin == lgTmax)
            D = float(parameters["lgDmin"])  # lgDmin (assuming lgDmin == lgDmax)  
      
            output_file = construct_output_filename(T, D)
            directory = parameters["DIR"].strip("'")
            file_path = directory + '/' + output_file
            log_print('Constructed file path:', file_path)

            # Write updated parameters and rerun the Fortran executable
            write_input_file(input_file, parameters)
            # run_fortran_program(executable, input_file)
            run_mforce(parameters)

            "kappa_e is read from the data provided by Mforce or can also be calculated"
            kappa_fm_mforce = True
            kappa_mf = read_kappa(parameters["DIR"].strip("'"))
    
            if (kappa_fm_mforce):
              kap_e = kappa_mf
            else:  
              kap_e = cgs.sigth/cgs.mass_p*(1.+Ihe*Yhe)/(1.+4.*Yhe)
            
            gamma_e = kap_e*lum/(4.*np.pi*cgs.G*M_star*cgs.c)
                       
            if (gamma_e >= 1):
              log_print('gamma_e > 1', gamma_e, kap_e)
              sys.exit(0) 
        
            cgas = np.sqrt(cgs.kb * T_eff / (mu * cgs.mass_p))
            v_esc = np.sqrt(2.0 * cgs.G * M_star / R_star * (1.0 - gamma_e))
            rat = v_esc / cgas
            phi_cook = 3.0 * rat**(0.3 * (0.36 + np.log10(rat)))
    
        
            qbar, alpha, q0, lgt_filtered = fit_data(file_path, t_cri)
            
            plot_fit(file_path, alpha, q0, qbar, iteration, t_cri, random_subdir)
            
            mdot_old = mdot
        
            mdot, cut = cak_massloss(lum, qbar, q0, alpha, gamma_e, 1./rat)
            mdot = max(mdot,1.e-16*cgs.Msun/cgs.year)

            v_cri = (phi_cook/(1-alpha))**0.5
            rho_target = mdot / (4 * np.pi * v_cri * cgas * R_star**2)
            
            t_cri = kap_e*cgs.c*rho_target/(v_cri*cgas/R_star)
            
            rel_rho = 1. - rho_target / rho
            rel_mdot = 1. - mdot / mdot_old
           
            if np.isnan(rho_target):
               raise ValueError('NaN in rho')
             
            it_num.append(iteration)
            mdot_num.append(mdot)
            qbar_num.append(qbar)
            alpha_num.append(alpha)
            q0_num.append(q0)
            eps_num.append(np.abs(rel_mdot))
            delrho_num.append(np.abs(rel_rho))

            mdot_lim = gamma_e*(1.+qbar)
            log_print(f"Iteration {iteration}, rho_target={np.log10(rho_target)}, rho={np.log10(rho)}")
            log_print(f"gamma_e*(1+qbar) = {mdot_lim}")
            #JS-Dec 16, want to see what's happening to convergence

            log_print(f"rel_mdot = {rel_mdot}, rel_rho = {rel_rho}")
            log_print(f"kappa_e = {kap_e}, Gamma_e = {gamma_e}, vesc = {v_esc}, rat = {rat}, phi_cook = {phi_cook}")
            log_print(f"Qbar = {qbar}, alpha = {alpha}, Q0 = {q0}")
            log_print(f"t_crit = {t_cri}")
            log_print(f"density = {rho}")
            log_print(f"Mass loss rate = {mdot*cgs.year/cgs.Msun}")
            rho = rho_target
        
            # The metallicity should be the actual Zfrac
            log_print(f"Zmass = {Z_star}")
            log_print(f"Zscale = {Z_scale}")
            
            z_metal = parameters["Z_mass"].strip("'")
            z_metal = z_metal.replace('d', 'e')
            z_metal = float(z_metal)
            
            "Different mass loss prescriptions"
            "The rescaled ones are after calculating the enhacements"
            "Otherwise using the scaling metallicity - e.g. 0.5 for LMC, 0.2 for SMC etc"
                   
            # prescriptions - rescaled
            bjor_mdot_scaled = bjorklund(lum,M_star,T_eff,gamma_e,z_metal)
            vink_mdot_scaled = Vink(T_eff,lum,M_star,gamma_e,z_metal)
            kriticka_mdot_scaled = Kriticka(lum,T_eff,z_metal)
                        
            # prescriptions - no rescaling
            bjor_mdot = bjorklund(lum,M_star,T_eff,gamma_e,Z_scale)
            vink_mdot = Vink(T_eff,lum,M_star,gamma_e,Z_scale)
            kriticka_mdot = Kriticka(lum,T_eff,Z_scale)
            
            log_print(f"Vink scaled = {vink_mdot_scaled}")
            log_print(f"Bjoklund scaled = {bjor_mdot_scaled}")
            log_print(f"Kriticka scaled = {kriticka_mdot_scaled}")
            
            log_print(f"Vink = {vink_mdot}")
            log_print(f"Bjoklund = {bjor_mdot}")
            log_print(f"Kriticka = {kriticka_mdot}")
            
            log_print("----------x----------x----------x----------x----------")

            # Plotting some stuffs

            fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True)
            # Flatten axes for easy indexing
            axes = axes.flatten()
            axes[0].plot(it_num, np.log10([mdot * cgs.year / cgs.Msun for mdot in mdot_num]), marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k', linestyle='--', linewidth=2)
            axes[1].plot(it_num, qbar_num, marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k', linestyle='--', linewidth=2)
            axes[2].plot(it_num, alpha_num, marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k', linestyle='--', linewidth=2)
            axes[3].plot(it_num, q0_num, marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k', linestyle='--', linewidth=2)
            axes[4].plot(it_num, np.log10(eps_num), marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k', linestyle='--', linewidth=2)
            axes[5].plot(it_num, np.log10(delrho_num), marker="D", markersize=10, markerfacecolor='gray', markeredgecolor='k', color='k', linestyle='--', linewidth=2)
            

            axes[0].set_ylabel(r"$\dot{M} [M_\odot/yr]$")
            axes[1].set_ylabel(r"$\bar{Q}$")
            axes[2].set_ylabel(r"$\alpha$")
            axes[3].set_ylabel(r"$Q_0$")            
            axes[4].set_ylabel(r"log $\epsilon_{\dot{M}}$")
            axes[5].set_ylabel(r"log $\epsilon_{\rho}$")
            # Set x-label for the bottom row only
            for ax in axes[-2:]:
                ax.set_xlabel("Iteration")
            for ax in axes:
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(useOffset=False, style='plain', axis='both') 
            #axes[2].set_xlabel("iteration")
            plt.tight_layout()
            plt.savefig(f"{random_subdir}/sim_log.png")

            
           #JS-JAN 2025 - imposing lower limit for validity of line-driven mass loss
            if iteration < 3:
                continue
            
            if iteration >= 3 and mdot_lim < 1:
                log_print("gamma_e*(1+qbar)<1")
                log_print("line-driven mass-loss rate not possible")
                log_print("Try to increase gamma_e or qbar (e.g. by higher L/M or higher Z)")
                log_print(f"{gamma_e}, {qbar}, {mdot_lim}")
                log_print("----------x----------x----------x----------x----------")
                return str(random_subdir) 
                       
            if alpha > 0.95:
                log_print("WARNING!! Alpha very high, approaching diverging limit")
                log_print(str(random_subdir))
                return str(random_subdir)
            
            if 3 <= iteration < 8 and np.abs(rel_rho) <= tolerance and np.abs(rel_mdot) <= tolerance:
                log_print("Final values (mdot, Qbar, alpha, q0):")
                log_print(mdot * cgs.year / cgs.Msun, qbar, alpha, q0)
                log_print(str(random_subdir))
                # return mdot*cgs.year/cgs.Msun, qbar, alpha, q0, t_cri, v_cri*cgas/1.e5, rho,
                return str(random_subdir)
            
            if iteration >= 8 and np.abs(rel_rho) < 1.e-2 and np.abs(rel_mdot) < 1.e-2:
                log_print("Final values (mdot, Qbar, alpha, Q0):")
                log_print(mdot * cgs.year / cgs.Msun, qbar, alpha, q0)
                log_print(str(random_subdir))
                return str(random_subdir)
            
            if iteration == max_iterations and  np.abs(rel_rho) > 1.e-2 and np.abs(rel_mdot) > 1.e-2:
                log_print("Not yet converged")
                log_print(str(random_subdir))
                return str(random_subdir)


if __name__ == "__main__":

    lum = float(sys.argv[1])
    T_eff = float(sys.argv[2])
    M_star = float(sys.argv[3])
    Z_star = float(sys.argv[4])
    Z_scale = float(sys.argv[5])
    Yhel = float(sys.argv[6])
    random_subdir = str(sys.argv[7])
    
    main(lum, T_eff, M_star, Z_star, Z_scale, Yhel, random_subdir)