from flask import Flask, request, jsonify, render_template
import subprocess
import json
import os
import threading
import smtplib
import shutil
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

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

"Setting up an SMTPS server, and recording the sender email, password"
SMTP_SERVER = "smtps.kuleuven.be"
SMTP_PORT = 587  
SENDER_EMAIL = "dwaipayan.debnath@kuleuven.be"
SENDER_PASSWORD = "Lampard10"

def calculate_metallicity(number_abundances):
    "Calculates the actual metallicty from the number "
    " abundances input by the user"
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

def send_email(recipient_email, subject, body, attachment_path=None):
    "This subroutine handles emailing to the user, with optional attachment"
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    # Attach file if provided
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
        msg.attach(part)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {str(e)}")

def process_computation(lum, teff, mstar, zscale, zstar, helium_abundance, abundances, recipient_email):
    "The input abundances are written out in the txt file"
    "This is later used by Mforce-LTE for it's calculations"
    "RUNS mcak_explore by py subprocesses"
    try:
        abundance_filename = os.path.join(DATA_DIR, "number_abundance")
        with open(abundance_filename, "w") as f:
            for i, (element, value) in enumerate(abundances.items(), start=1):
                f.write(f"{i:2d}  '{element.upper():2s}'   {value:.14f}\n")
        # result = subprocess.Popen(
        #     ["python3", "-c", "print(test)"],
        #     stdout=subprocess.PIPE, stderr=subprocess.PIPE
        # )
        result = subprocess.Popen(
            ["python3", "mcak_explore.py", str(lum), str(teff), str(mstar), str(zstar), str(zscale), str(helium_abundance)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(result.communicate())
        

        if result.returncode == 0:
            print("mcak_explore.py executed successfully")
        else:
            print(f"Computation error: {result.stderr}")
            #send_email(recipient_email, "Computation Error", f"Error occurred:\n{result.stderr}")
            return  # Stop execution if there's an error

        # Zip the results directory
        output_dir = f"./tmp/{result}" 
        zip_filename = os.path.join(output_dir, f"{result}.zip")
        print(output_dir, zip_filename)
        shutil.make_archive(zip_filename.replace(".zip", ""), 'zip', output_dir)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    # Send email with the ZIP file
    send_email(recipient_email, "LIME Computation Results", "Attached is the LIME output", zip_filename)
    
            
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/process_data', methods=['POST'])
def process_data():
    "Handles the communication with index.html"
    try:
        data = request.json
        print("Received data:", data)  # Debugging line

        # Ensure required parameters exist and are valid
        luminosity = float(data.get("luminosity", 0.0))
        teff = float(data.get("teff", 0.0))
        mstar = float(data.get("mstar", 0.0))
        zscale = float(data.get("zscale", 0.0))
        abundances = data.get("abundances", {})  # Keep this as a dictionary
        helium_abundance = float(abundances.get("HE", 0.0))  # Extract HE safely
        user_email = data.get("email", "").strip()

        if not user_email:
            return jsonify({"error": "Email is required"}), 400

        if not abundances:
            return jsonify({"error": "Abundances data is missing"}), 400

        # Calculate Zstar correctly
        zstar = calculate_metallicity(abundances)

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
