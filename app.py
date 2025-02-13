from flask import Flask, request, jsonify, render_template
import subprocess
import numpy as np
import json
import os
import threading
import shutil
import re
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from PIL import Image

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
        
        pdf_filename = f"./{output_dir}/report.pdf"
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
        
        print(table_data)
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

        body = load_email_body('./mailing/mail_template.j2')
        
        subprocess.run(["python3", "./mailing/mailer.py", "--t", recipient_email, "--s", "LIME Computation Results", "--b", body, "--a", pdf_filename])

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
