from flask import Flask, request, jsonify, render_template, send_file, url_for
import subprocess
import numpy as np
import json
import os
import threading
import shutil
import re
from mcak_explore import main as mcak_main
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from PIL import Image
import pandas as pd
import tempfile
import random
import zipfile
import time


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
    N_C = (mass_C / ATOMIC_MASSES['C']) / total_num_abun

    if N_H == 0.0:
        NHe = N_He/N_He
    elif N_H == 0.0 and N_He == 0.0:
        NHe = N_He/N_C    
    else :
        NHe = N_He/N_H    

    return NHe

def load_email_body(filename):
    with open(filename, 'r') as file:
        return file.read()

def process_computation(lum, teff, mstar, zscale, zstar, helium_abundance, abundances, recipient_email, pdf_name, pointer, batch_output_dir):
    """Runs mcak_explore and emails the results"""
    # Making a temporary file
    #base_tmp_dir = f"{os.path.abspath(os.getcwd())}/tmp"
    #os.makedirs(base_tmp_dir, exist_ok=True)
    #random_subdir = tempfile.mkdtemp(dir=base_tmp_dir)
    #os.makedirs(random_subdir, exist_ok=True) 
    #massabun_loc = os.path.join(random_subdir, "output")
    #os.makedirs(massabun_loc, exist_ok=True)

    try:
        output_dir = os.path.join(batch_output_dir, pdf_name)
        os.makedirs(output_dir, exist_ok=True)
        massabun_loc = os.path.join(output_dir, "output")
        os.makedirs(massabun_loc, exist_ok=True)
        abundance_filename = os.path.join(massabun_loc, "mass_abundance")
        with open(abundance_filename, "w") as f:
            for i, (element, value) in enumerate(abundances.items(), start=1):
                f.write(f"{i:2d}  '{element.upper():2s}'   {value:.14f}\n")

        # Run computation
        # result = subprocess.run(
        #     ["python3", "mcak_explore.py", str(lum), str(teff), str(mstar), str(zstar), str(zscale), str(helium_abundance)])#,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # output = result.stdout.decode().strip()

        generated_file = mcak_main(lum, teff, mstar,zstar, zscale, helium_abundance, ouptut_subdir)

        # output_lines = output.splitlines()
        # generated_file = output_lines[-1]
        print(generated_file)

        if generated_file is None:
            print("Failed somewhere! Sorry!")
            # print(f"Computation error: {result.stderr}")
            return  
        
        # Get the directory from the generated file
        #output_dir = generated_file.strip()
        #output_filename = os.path.basename(output_dir)
        pdf_filename = os.path.join(output_dir, f"{pdf_name}.pdf")
        
        
        pdf_filename = f"{output_dir}/{pdf_name}"
        figures_list = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")]
        simlog_path = os.path.join(output_dir, "simlog.txt")

        c = canvas.Canvas(pdf_filename, pagesize=letter)
        page_width, page_height = letter  # Letter page size (8.5 x 11 inches)
        
        # --- Add Logo and Title on First Page ---
        logo_path = "./static/logo_2.png"  # Adjust path as needed
        title = "LIME Results"
        
        if os.path.exists(logo_path):
            c.drawImage(logo_path, 50, page_height - 150, width=120, height=120)
        
        c.setFont("Helvetica-Bold", 30)
        c.drawString(220, page_height - 150, title)

        # Update table_data to a transposed format
        table_data = [
            ("Mass Loss Rate", "Qbar", "Alpha", "Q0"),  # Column headers
        ]
        
        # Read values from the simlog.txt and ensure the correct number of elements are available
        with open(simlog_path, "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                last_values = lines[-2].split()
                if len(last_values) < 5:
                    wrmdot = float(last_values[0].strip("'"))
                    wrqbar = float(last_values[1].strip("'"))
                    wralp = float(last_values[2].strip("'"))
                    wrq0 = float(last_values[3].strip("'"))
                    table_data.append((f"{wrmdot:.3e}", f"{wrqbar:.2e}",f"{wralp:.2e}",f"{wrq0:.2e}"))
        
        # Draw the table (Ensure it's horizontal)
        c.setFont("Helvetica", 16)
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        # Wrap and draw the table on the PDF
        table.wrapOn(c, page_width, page_height)
        table.drawOn(c, 70, page_height - 250)
        
        # --- 1. Display "sim_log.png" on its own full page ---
        sim_log_image_path = os.path.join(output_dir, "sim_log.png")
        if sim_log_image_path in figures_list:
            c.showPage()  # Ensure a new page
            img = Image.open(sim_log_image_path)
            
            # Scale the image to fit the full page
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            new_width = page_width - 50  # Leave some margin
            new_height = new_width / aspect_ratio  # Keep aspect ratio
        
            if new_height > page_height - 50:  # If it exceeds page height, scale differently
                new_height = page_height - 50
                new_width = new_height * aspect_ratio
        
            x_position = (page_width - new_width) / 2  # Center horizontally
            y_position = (page_height - new_height) / 2  # Center vertically
        
            c.drawImage(sim_log_image_path, x_position, y_position, width=new_width, height=new_height)
        
            # Add label
            c.setFont("Helvetica", 14)
            des = "Different quantities (check legends) as function of iteration number until convergence has been reached. The bottom panels show relative differences in mass loss and density between two successive iterations."
            
            text_x = 50
            text_y = 60 

            # Wrap text if it's too long
            wrapped_text = simpleSplit(des, "Helvetica", 14, page_width - 100)
            
            # Draw wrapped text
            for line in wrapped_text:
                c.drawString(text_x, text_y, line)
                text_y -= 16  # Move down for next line

            # Remove it from the list to avoid duplication
            figures_list.remove(sim_log_image_path)
        
        # --- 2. Arrange Remaining Figures in a 2x2 Grid (4 per page) ---
        images_per_row = 2
        images_per_col = 2
        images_per_page = images_per_row * images_per_col
        
        image_width = (page_width - 100) / images_per_row  # Leave margin
        image_height = (page_height - 400) / images_per_col  # Leave margin
        
        x_start = 50  # Margin from left
        y_start = page_height - 300  # Start from the top (leave space for labels)
        
        for i, fig_path in enumerate(figures_list):
            if i % images_per_page == 0:
                c.showPage()  # New page for every 4 images
            
            row = (i % images_per_page) // images_per_row
            col = (i % images_per_page) % images_per_row
        
            x_position = x_start + col * image_width
            y_position = y_start - row * image_height
        
            c.drawImage(fig_path, x_position, y_position, width=image_width, height=image_height)
        
            # Add label below each image
            c.setFont("Helvetica", 10)
            c.drawString(x_position, y_position - 15, f"Figure {i + 2}: {os.path.basename(fig_path)}")
        
        # --- 3. Add Simulation Log (simlog.txt) on a New Page ---
        if os.path.exists(simlog_path):
            c.showPage()  # New page for text
            c.setFont("Helvetica", 12)
            y_position = page_height - 50  # Start from the top
        
            with open(simlog_path, "r") as f:
                log_text = f.readlines()
        
            for line in log_text:
                if y_position < 50:  # If out of space, start a new page
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_position = page_height - 50
        
                c.drawString(50, y_position, line.strip())  # Add text line
                y_position -= 12  # Move down
        
        c.save()
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/process_data', methods=['POST'])
def process_data():
    """Handles the communication with index.html"""
    try:
        data = request.json
        print("Received data:", data)

        # Extract parameters
        luminosity = float(data.get("luminosity", 0.0))
        teff = float(data.get("teff", 0.0))
        mstar = float(data.get("mstar", 0.0))
        zscale = float(data.get("zscale", 0.0))
        abundances = data.get("abundances", {})

        if not abundances:
            return jsonify({"error": "Abundances data is missing"}), 400

        # Calculate values
        zstar = calculate_metallicity_massb(abundances)
        helium_abundance = He_number_abundance(abundances)

        # Create a temporary directory
        base_tmp_dir = "./tmp"
        os.makedirs(base_tmp_dir, exist_ok=True)
        session_tmp_dir = tempfile.mkdtemp(dir=base_tmp_dir)

        pdf_name = "result"
        output_dir = os.path.join(session_tmp_dir, pdf_name)  
        os.makedirs(output_dir, exist_ok=True)  
        pdf_path = os.path.join(output_dir, f"{pdf_name}.pdf")  

        # Run computation in a separate thread
        computation_thread = threading.Thread(
            target=process_computation,
            args=(luminosity, teff, mstar, zscale, zstar, helium_abundance, abundances, "", pdf_name, -1, session_tmp_dir)
        )
        computation_thread.start()
        computation_thread.join()  # Wait for completion before proceeding

        # Check if the PDF exists at the correct path
        if not os.path.exists(pdf_path):
            return jsonify({"error": f"PDF generation failed. Expected at {pdf_path}"}), 500

        # Schedule cleanup
        threading.Timer(600, shutil.rmtree, args=[session_tmp_dir], kwargs={"ignore_errors": True}).start()

        # Generate a downloadable link
        relative_session_id = os.path.relpath(session_tmp_dir, base_tmp_dir)

        pdf_url = url_for("download_temp_file", session_id=relative_session_id, filename=f"{pdf_name}/result.pdf", _external=True)
        
        print(pdf_url)
        return jsonify({"message": "Computation complete", "download_url": pdf_url}), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid numerical input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/tmp/<session_id>/<path:filename>')
def download_temp_file(session_id, filename):
    session_tmp_dir = os.path.join("./tmp", session_id)  
    file_path = os.path.abspath(os.path.join(session_tmp_dir, filename))  

    # Security check: Prevent directory traversal
    if not file_path.startswith(os.path.abspath(session_tmp_dir)):
        return jsonify({"error": "Invalid file path."}), 403

    # Debugging: Print expected path
    print(f"Looking for file at: {file_path}")

    if not os.path.exists(file_path):
        print(f"File NOT FOUND: {file_path}")  # Debugging output
        return jsonify({"error": f"File {filename} not found"}), 404

    return send_file(file_path, mimetype='application/pdf')

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    """Handles CSV file upload and batch computation"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        df = pd.read_csv(file)
        num_rows = len(df)

        # **Check the number of rows in CSV**
        if num_rows > 100:
            return jsonify({
                "error": "Too many entries! Please reduce the number of stars to 100 or split the CSV into smaller parts."
            }), 400

        # Create a single batch directory for all results
        base_dir = "./tmp"
        os.makedirs(base_dir, exist_ok=True)
        batch_output_dir = tempfile.mkdtemp(dir=base_dir)
        os.makedirs(batch_output_dir, exist_ok=True)

        print(f"Batch directory created: {batch_output_dir}")  # Debugging
        
        user_email = request.form.get("email", "").strip()
        if not user_email:
            return jsonify({"error": "Email is required"}), 400

        # **Return success response immediately**
        response_message = {
            "message": "CSV uploaded successfully. Your calculations will be processed and emailed to you shortly."
        }

        def process_batch():
            """Function to process the batch asynchronously"""
            try:
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

                computation_threads = []
                pdf_paths = []
                results_data = []  

                for index, row in df.iterrows():
                    pdf_name = str(row["name"])
                    lum = float(row["luminosity"])
                    teff = float(row["teff"])
                    mstar = float(row["mstar"])
                    zscale = float(row["zscale"])

                    # Create subdirectory inside batch directory
                    result_dir = os.path.join(batch_output_dir, pdf_name)
                    os.makedirs(result_dir, exist_ok=True)
                    pdf_path = os.path.join(result_dir, f"{pdf_name}.pdf")
                    pdf_paths.append(pdf_path)

                    abundances = {}
                    for element, default_value in default_abundances.items():
                        if element in ["H", "HE"]:
                            abundances[element] = float(row[element]) if element in row and not pd.isna(row[element]) else default_value
                        else:
                            abundances[element] = float(row[element]) if element in row and not pd.isna(row[element]) else default_value * zscale

                    zstar = calculate_metallicity_massb(abundances)
                    helium_abundance = He_number_abundance(abundances)

                    pointer = -1

                    computation_thread = threading.Thread(
                        target=process_computation,
                        args=(lum, teff, mstar, zscale, zstar, helium_abundance, abundances, user_email, pdf_name, pointer, batch_output_dir)
                    )
                    computation_threads.append(computation_thread)
                    computation_thread.start()

                # Wait for all threads to finish
                for thread in computation_threads:
                    thread.join()

                # Collect results from each computation
                for index, row in df.iterrows():
                    pdf_name = str(row["name"])
                    result_dir = os.path.join(batch_output_dir, pdf_name)
                    simlog_path = os.path.join(result_dir, "simlog.txt")

                    if os.path.exists(simlog_path):
                        with open(simlog_path, "r") as f:
                            lines = f.readlines()
                            if len(lines) >= 2:
                                last_values = lines[-2].split()
                                if len(last_values) >= 4:
                                    wrmdot = float(last_values[0].strip("'"))
                                    wrqbar = float(last_values[1].strip("'"))
                                    wralp = float(last_values[2].strip("'"))
                                    wrq0 = float(last_values[3].strip("'"))
                                else:
                                    wrmdot, wrqbar, wralp, wrq0 = None, None, None, None
                            else:
                                wrmdot, wrqbar, wralp, wrq0 = None, None, None, None
                    else:
                        wrmdot, wrqbar, wralp, wrq0 = None, None, None, None

                    results_data.append({
                        "Name": pdf_name,
                        "Luminosity": row["luminosity"],
                        "Teff": row["teff"],
                        "Mstar": row["mstar"],
                        "Zscale": row["zscale"],
                        "Mass Loss Rate": wrmdot,
                        "Qbar": wrqbar,
                        "Alpha": wralp,
                        "Q0": wrq0
                    })

                # Save results to CSV file
                results_csv_path = os.path.join(batch_output_dir, "results.csv")
                results_df = pd.DataFrame(results_data)
                results_df.to_csv(results_csv_path, index=False)

                # **Create ZIP file with results**
                final_zip_path = os.path.join(batch_output_dir, "final_results.zip")
                with zipfile.ZipFile(final_zip_path, 'w') as zipf:
                    if num_rows > 20:
                        # **Only add CSV for large files**
                        zipf.write(results_csv_path, os.path.basename(results_csv_path))
                    else:
                        # **Include both CSV and PDFs for small files**
                        for pdf_path in pdf_paths:
                            if os.path.exists(pdf_path):
                                zipf.write(pdf_path, os.path.relpath(pdf_path, batch_output_dir))
                        zipf.write(results_csv_path, os.path.basename(results_csv_path))

                # Send email with ZIP
                body = load_email_body('./mailing/mail_template.j2')
                subprocess.run(["python3", "./mailing/mailer.py", "--t", user_email, "--s", "LIME Computation Results", "--b", body, "--a", final_zip_path])

            except Exception as e:
                print(f"Unexpected error in batch processing: {str(e)}")

        # **Start computation in a new thread**
        batch_thread = threading.Thread(target=process_batch)
        batch_thread.start()

        return jsonify(response_message), 200

    except Exception as e:
        return jsonify({"error": f"Unexpected error in upload_csv: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
