from flask import Flask, request, jsonify, render_template
import subprocess
import json
import os
import threading
import shutil
import re

app = Flask(__name__, static_folder='static', template_folder='templates')

MFORCE_DIR = os.getenv("MFORCE_DIR", ".")
DATA_DIR = os.path.join(MFORCE_DIR, "DATA")
os.makedirs(DATA_DIR, exist_ok=True)  

# Atomic masses for elements
ATOMIC_MASSES = {
    'H': 1.008, 'HE': 4.0026, 'LI': 6.941, 'BE': 9.012, 'B': 10.811, 'C': 12.011,
    'N': 14.007, 'O': 16.000, 'F': 18.998, 'NE': 20.180, 'NA': 22.990, 'MG': 24.305,
    'AL': 26.982, 'SI': 28.085, 'P': 30.974, 'S': 32.066, 'CL': 35.453, 'AR': 39.948,
    'K': 39.098, 'CA': 40.078, 'SC': 44.956, 'TI': 47.880, 'V': 50.941, 'CR': 51.996,
    'MN': 54.938, 'FE': 55.847, 'CO': 58.933, 'NI': 58.690, 'CU': 63.546, 'ZN': 65.390
}

def calculate_metallicity_massb(mass_abundances):
    """Calculates the actual metallicity from the number abundances input by the user"""
    metals = {e for e in ATOMIC_MASSES if e not in {'H', 'HE'}}
    metallicity = sum(
        mass_abundances[element]
        for element in metals if element in mass_abundances
    )
    
    return metallicity

def calculate_metallicity(number_abundances):
    """Calculates the actual metallicity from the number abundances input by the user"""
    total_mass_abundance = sum(
        number_abundances[element] * ATOMIC_MASSES[element]
        for element in number_abundances if element in ATOMIC_MASSES
    )
    
    metals = {e for e in ATOMIC_MASSES if e not in {'H', 'HE'}}
    metallicity = sum(
        number_abundances[element] * ATOMIC_MASSES[element] / total_mass_abundance
        for element in metals if element in number_abundances
    )
    
    return metallicity

def He_number_abundance(mass_abundances):
    """
    Compute the number abundance of helium from mass abundances.
    This is relative to the Hydrogen abundances.
    
    :return: Number abundance of helium (relative to all elements)
    """
    
    mass_H = mass_abundances.get('H', 0.0)
    mass_He = mass_abundances.get('HE', 0.0)
    mass_C = mass_abundances.get('C', 0.0)
    
    total_num_abun = sum(mass_abundances[element] / ATOMIC_MASSES[element] for element in mass_abundances)

    N_H = (mass_H / ATOMIC_MASSES['H']) / total_num_abun
    N_He = (mass_He / ATOMIC_MASSES['HE']) / total_num_abun
    N_C = (mass_He / ATOMIC_MASSES['C']) / total_num_abun

    if N_H == 0.0:
        NHe = N_He/N_He
    else :
        NHe = N_He/N_C     

    return NHe

def load_email_body(filename):
    with open(filename, 'r') as file:
        return file.read()

def process_computation(lum, teff, mstar, zscale, zstar, helium_abundance, abundances, recipient_email):
    """Runs mcak_explore and emails the results"""
    try:
        abundance_filename = os.path.join(DATA_DIR, "mass_abundance")
        with open(abundance_filename, "w") as f:
            for i, (element, value) in enumerate(abundances.items(), start=1):
                f.write(f"{i:2d}  '{element.upper():2s}'   {value:.14f}\n")

        # Run computation
        result = subprocess.run(
            ["python3", "mcak_explore.py", str(lum), str(teff), str(mstar), str(zstar), str(zscale), str(helium_abundance)],stdout=subprocess.PIPE, stderr=subprocess.PIPE)         
        output = result.stdout.decode().strip()

        output_lines = output.splitlines() 
        generated_file = output_lines[-1] 
        print(generated_file)

        if result.returncode != 0:
            print(f"Computation error: {result.stderr}")
            return  
        
        # Get the directory from the generated file
        output_dir = generated_file.strip()
        output_filename = os.path.basename(output_dir)
        
        tmp_dir = './tmp'
        
        # Create a zip file of the contents of output_dir, not including the output_dir itself
        zip_filename = os.path.join(tmp_dir, f"{output_filename}.zip")
        shutil.make_archive(zip_filename.replace(".zip", ""), 'zip', output_dir)
        

    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    body = load_email_body('./mailing/mail_template.j2')
    
    subprocess.run(["python3", "./mailing/mailer.py", "--t", recipient_email, "--s", "LIME Computation Results", "--b", body, "--a", zip_filename])

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/process_data', methods=['POST'])
def process_data():
    """Handles the communication with index.html"""
    try:
        data = request.json
        print("Received data:", data)  

        # Ensure required parameters exist and are valid
        luminosity = float(data.get("luminosity", 0.0))
        teff = float(data.get("teff", 0.0))
        mstar = float(data.get("mstar", 0.0))
        zscale = float(data.get("zscale", 0.0))
        abundances = data.get("abundances", {})    
        user_email = data.get("email", "").strip()

        if not user_email:
            return jsonify({"error": "Email is required"}), 400

        if not abundances:
            return jsonify({"error": "Abundances data is missing"}), 400

        # Calculate Zstar correctly from mass abundances
        zstar = calculate_metallicity_massb(abundances)
        
        
        # calculating the number abundance of helium from the mass abundance
        helium_abundance = He_number_abundance(abundances)
        print('helium',helium_abundance)

        computation_thread = threading.Thread(
            target=process_computation,
            args=(luminosity, teff, mstar, zscale, zstar, helium_abundance, abundances, user_email)
        )
        computation_thread.start()

        return jsonify({"message": "Data submitted successfully. You will receive an email when the computation is complete."}), 200
    except ValueError as e:
        return jsonify({"error": f"Invalid numerical input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
