from flask import Flask, request, jsonify, render_template, send_file, url_for
import subprocess
import numpy as np
import os
import threading
import shutil
from mcak_explore import main as mcak_main
from mcak_explore import DUMMY_RESULTS, color
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors
from PIL import Image
import pandas as pd
import tempfile
from jinja2 import Template
import csv
from celery import Celery, Task, shared_task
import io


def celery_init_app(app: Flask) -> Celery:
    """Initialization of the Celery app
    Taken from https://flask.palletsprojects.com/en/stable/patterns/celery/"""
    class FlaskTask(Task):
        def __call__(self, *args: object, **kwargs: object) -> object:
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    celery_app.conf.worker_concurrency = 4
    app.extensions["celery"] = celery_app
    return celery_app


# Initialize the website app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Use redis as broker for the queueing system.
app.config.from_mapping(
    CELERY=dict(
        broker_url="redis://localhost",
        result_backend="redis://localhost",
        task_ignore_result=True,
    ),
)

# Server property setup
app.config["SERVER_NAME"] = "0.0.0.0:8000"
app.config["PREFERRED_URL_SCHEME"] = "http"
app.config["APPLICATION_ROOT"] = ""

# Start Celery for queueing
celery = celery_init_app(app)

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
    metallicity = sum(mass_abundances[element] for element in metals if element in mass_abundances)
    return metallicity


def He_number_abundance(mass_abundances):
    """
    Compute the number abundance of helium from mass abundances.
    This is relative to the Hydrogen abundances.
    
    :return: Number abundance of helium (relative to all elements)
    """
    mass_H = mass_abundances.get('H', 0.0)
    mass_He = mass_abundances.get('HE', 0.0)
    
    total_num_abun = sum(mass_abundances[element] / ATOMIC_MASSES[element] for element in mass_abundances)

    N_H = (mass_H / ATOMIC_MASSES['H']) / total_num_abun
    N_He = (mass_He / ATOMIC_MASSES['HE']) / total_num_abun

    if N_H == 0.0 and N_He == 0.0:
        NHe = 1e-4    
    else:
        NHe = N_He/N_H    
    return NHe


def load_email_body(filename):
    with open(filename, 'r') as file:
        return file.read()


def load_dyn_email(filename, context):
    """Load and render email body from Jinja2 template."""
    with open(filename, 'r') as file:
        template = Template(file.read())
    return template.render(context)


def process_computation(lum, teff, mstar, zscale, zstar, helium_abundance, abundances, pdf_name,
                        batch_output_dir, expert_mode, does_plot):
    """Runs mcak_explore and generates a pdf with the results if desired. """
    try:
        output_dir = os.path.join(batch_output_dir, pdf_name)
        os.makedirs(output_dir, exist_ok=True)
        massabun_loc = os.path.join(output_dir, "output")
        os.makedirs(massabun_loc, exist_ok=True)
        abundance_filename = os.path.join(massabun_loc, "mass_abundance")
        
        with open(abundance_filename, "w") as f:
            for i, (element, value) in enumerate(abundances.items(), start=1):
                f.write(f"{i:2d}  '{element.upper():2s}'   {value:.14f}\n")

        generated_file, results_dict = mcak_main(lum, teff, mstar, zstar, zscale, helium_abundance, output_dir, does_plot)

        if results_dict["fail"]:
            failure_reason = results_dict["fail_reason"]
            print(f"{color.RED}Simulation failed: {failure_reason}{color.END}")
            pdf_filename = os.path.join(output_dir, f"{pdf_name}.pdf")
            c = canvas.Canvas(pdf_filename, pagesize=letter)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(220, 700, "Simulation Failed")
            c.setFont("Helvetica", 14)
            c.drawString(100, 650, "Reason for failure:")
            wrapped_text = simpleSplit(failure_reason, "Helvetica", 12, 400)
            y_pos = 620
            for line in wrapped_text:
                c.drawString(120, y_pos, line)
                y_pos -= 20
            c.save()
            # Leave it at this if the calculation failed
            return results_dict

        # make some diagnostic plots and informative tables for the pdf output.
        if does_plot == True :
            pdf_filename = os.path.join(output_dir, f"{pdf_name}.pdf")
            figures_list = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".png")]

            # Initialize
            c = canvas.Canvas(pdf_filename, pagesize=letter)
            page_width, page_height = letter
    
            logo_path = "./static/logo_2.png"
            title = "LIME Results"
            
            if os.path.exists(logo_path):
                c.drawImage(logo_path, 50, page_height - 150, width=120, height=120, preserveAspectRatio=True, anchor='c')

            # Add a warning at the top of the pdf file if there is a potential issue
            warning_path = "./static/warning.png"
            if results_dict["warning"]:
                c.setFont("Helvetica", 10)  
                c.drawString(130, page_height - 15, results_dict["warning_message"])
                if os.path.exists(warning_path):
                    c.drawImage(warning_path, 113, page_height - 17.5, width=15, height=15, preserveAspectRatio=True, anchor='c')

            c.setFont("Helvetica-Bold", 30)
            c.drawString(220, page_height - 150, title)
    
            table_data = [
                ("Input Parameter", "Value"),
                ("Luminosity [solar luminosity]", f"{lum:.1f}"),
                ("Stellar Mass [solar mass]", f"{mstar:.1f}"),
                ("Eddington Ratio ", "{:.2f}".format(results_dict["kappa_e"])),
                ("Stellar Radius [solar radius] ","{:.2f}".format(results_dict["R_star"])),
                ("log g ","{:.2f}".format(results_dict["log_g"])),
                ("Effective Temperature [K]", f"{teff:.1f}"),
                ("Z star (Calculated)", f"{zstar:.3e}")]
            
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
            table.wrapOn(c, page_width, page_height)
            table.drawOn(c, 70, page_height - 350)

            # Make the results table
            table_data = [("Output", "Value"),
                          ("Mass loss rate [solar mass/year]", "{:.3e}".format(results_dict["mdot"])),
                          ("vinf [km/s]",  "{:.2f}".format(results_dict["vinf"])),
                          ("Electron scattering opacity ",  "{:.2f}".format(results_dict["kappa_e"])),
                          ("Critical depth", "{:.2f}".format(results_dict["t_crit"])),
                          ("Q bar", "{:.2f}".format(results_dict["Qbar"])),
                          ("alpha", "{:.2f}".format(results_dict["alpha"])),
                          ("Q0", "{:.2f}".format(results_dict["Q0"]))]

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
            table.wrapOn(c, page_width, page_height)
            table.drawOn(c, 70, page_height - 530)

            # If desired produce more information in the output
            if expert_mode:
                table_data = [("Extra Output", "Value"),
                              ("Z scaled to solar (input)", f"{zscale:.2e}"),
                              ("Globally fitted alpha", "{:.3f}".format(results_dict["alphag"])),
                              ("Locally fitted alpha", "{:.3f}".format(results_dict["alpha2"])),
                              ("Effective v escape [km/s]", "{:.2f}".format(results_dict["vesc"])),
                              ("Critical velocity", "{:.2f}".format(results_dict["v_crit"])),
                              ("Critical density", "{:.2e}".format(results_dict["density"]))]

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
                table.wrapOn(c, page_width, page_height)
                table.drawOn(c, 70, page_height - 690)

            # Abundances Table
            abundance_table_data = [("Element", "Abundance")] + [(el, f"{abundances[el]:.4e}") for el in abundances]
            abundance_table = Table(abundance_table_data, colWidths=[100, 150])
            abundance_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            abundance_table.wrapOn(c, page_width, page_height)
            abundance_table.drawOn(c, 300, page_height - 760)
    
            # --- 1. Display "sim_log.png" with a title ---
            sim_log_image_path = os.path.join(output_dir, "sim_log.png")
            if sim_log_image_path in figures_list:
                c.showPage()
                c.setFont("Helvetica-Bold", 30)
                c.drawString(150, page_height - 50, "Simulation Log Figures")
    
                img = Image.open(sim_log_image_path)
                img_width, img_height = img.size
                aspect_ratio = img_width / img_height
                new_width = page_width - 100
                new_height = new_width / aspect_ratio
    
                if new_height > page_height - 100:
                    new_height = page_height - 50
                    new_width = new_height * aspect_ratio
    
                x_position = (page_width - new_width) / 2
                y_position = (page_height - new_height) / 2
    
                c.drawImage(sim_log_image_path, x_position, y_position, width=new_width, height=new_height)
    
                c.setFont("Helvetica", 14)
                description = "This figure shows the evolution of various physical parameters over the iterations until convergence."
                text_x, text_y = 50, 80
    
                wrapped_text = simpleSplit(description, "Helvetica", 14, page_width - 100)
                for line in wrapped_text:
                    c.drawString(text_x, text_y, line)
                    text_y -= 16
    
                figures_list.remove(sim_log_image_path)
    
            # --- 2. Arrange Remaining Figures in Pages with Titles ---
            images_per_row, images_per_col = 2, 3
            images_per_page = images_per_row * images_per_col
            max_width = (page_width - 100) / images_per_row
            max_height = (page_height - 150) / images_per_col
            x_start, y_start = 50, page_height - 300
    
            for i, fig_path in enumerate(figures_list):
                if i % images_per_page == 0:
                    c.showPage()
    
                c.setFont("Helvetica-Bold", 30)
                c.drawString(140, page_height - 50, "Line force multiplier v t")
    
                row = (i % images_per_page) // images_per_row
                col = (i % images_per_page) % images_per_row
                x_position = x_start + col * max_width
                y_position = y_start - row * max_height
    
                # Load image to get actual dimensions
                with Image.open(fig_path) as img:
                    img_width, img_height = img.size
    
                # Compute scaling factor while maintaining aspect ratio
                scale_factor = min(max_width / img_width, max_height / img_height)
                new_width = img_width * scale_factor
                new_height = img_height * scale_factor
    
                # Center the image within its allocated space
                x_adjusted = x_position + (max_width - new_width) / 2
                y_adjusted = y_position + (max_height - new_height) / 2
    
                c.drawImage(fig_path, x_adjusted, y_adjusted, width=new_width, height=new_height)
    
                c.setFont("Helvetica", 10)

            c.save()  

        return results_dict

    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    print("Process Computation failed!")
    return DUMMY_RESULTS


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/process_data', methods=['POST'])
def start_process():
    """This starts the data processing task"""
    data = request.json

    # Check for issues before putting calculations in the queue
    teff = float(data.get("teff", 0.0))
    if teff > 60000 or teff < 15000:
        return jsonify({"error": "Temperature beyond current coverage"}), 400

    abundances = data.get("abundances", {})
    if not abundances:
        return jsonify({"error": "Abundances data is missing"}), 400

    task = process_data.apply_async(args=[data])
    return jsonify({"task_id": task.id}), 202


@shared_task(ignore_result=False)
def process_data(data):
    """Handles the communication with index.html"""
    try:
        print("Received data:", data)

        # Extract parameters
        luminosity = float(data.get("luminosity", 0.0))
        teff = float(data.get("teff", 0.0))

        mstar = float(data.get("mstar", 0.0))
        zscale = float(data.get("zscale", 0.0))
        abundances = data.get("abundances", {})
        
        recipient_email = data.get("email", "").strip()

        # To allow additional output:
        expert_mode = data.get("expert_mode", False)  # by default false
        
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

        # In individual calculations always make the verification plots
        does_plot = True

        # Run computation in a separate thread
        results_dict = process_computation(luminosity, teff, mstar, zscale, zstar, helium_abundance, abundances,
                                           pdf_name, session_tmp_dir, expert_mode, does_plot)
        # Check if the PDF exists at the correct path
        if not os.path.exists(pdf_path):
            with app.app_context():
                return {"error": f"PDF generation failed. Expected at {pdf_path}"}, 500

        # Schedule cleanup
        threading.Timer(600, shutil.rmtree, args=[session_tmp_dir], kwargs={"ignore_errors": True}).start()

        # Generate a downloadable link
        relative_session_id = os.path.relpath(session_tmp_dir, base_tmp_dir)

        pdf_url = url_for("download_temp_file", session_id=relative_session_id, filename=f"{pdf_name}/result.pdf", _external=True)

        # **Send Email if recipient email is provided**
        if recipient_email:
            if results_dict is None:
                email_context = {"Result": "Failed to calculate a model"}
            else:
                if not results_dict["fail"]:
                    email_context = {
                        "luminosity": f"{luminosity:.1f}",
                        "teff": f"{teff:.1f}",
                        "mstar": f"{mstar:.1f}",
                        "rstar": "{:.2f}".format(results_dict["R_star"]),
                        "logg": "{:.2f}".format(results_dict["log_g"]),
                        "zstar": "{:.3e}".format(results_dict["Zmass"]),
                        "mass_loss_rate": "{:.3e}".format(results_dict["mdot"]),
                        "qbar": "{:.0f}".format(results_dict["Qbar"]),
                        "alpha": "{:.2f}".format(results_dict["alpha"]),
                        "q0": "{:.0f}".format(results_dict["Q0"]),
                        "vinf": "{:.2e}".format(results_dict["vinf"]),}
                    email_body = load_dyn_email('./mailing/mail_dyn.j2', email_context)
                else:
                    email_context = {"Result": results_dict["fail_reason"]}
                    email_body = load_dyn_email('./mailing/fail_template.j2', email_context)

            # Send the email using mailer.py
            subprocess.run([
                "python3", "./mailing/mailer.py",
                "--t", recipient_email,
                "--s", "LIME Computation Results",
                "--b", email_body])

        with app.app_context():
            return {"message": "Computation complete", "download_url": pdf_url}, 200
    
    except ValueError as e:
        with app.app_context():
            return {"error": f"Invalid numerical input: {str(e)}"}, 400
    except Exception as e:
        with app.app_context():
            return {"error": f"Unexpected error: {str(e)}"}, 500


@app.route("/task_status/<task_id>", methods=["GET"])
def get_processing_status(task_id):
    """Check how the calculation is doing"""
    task = celery.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return jsonify({"status": task.state, "done": True, **task.result[0]}), task.result[1]
    else:
        return jsonify({"task_id": task.id, "status": task.state}), 202


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

    response = send_file(file_path, mimetype='application/pdf')
    
    def cleanup():
        shutil.rmtree(session_tmp_dir, ignore_errors=True)
    
    threading.Timer(5, cleanup).start()  

    return response


@app.route('/upload_csv', methods=['POST'])
def start_upload_csv():
    """This starts the processing of the csv file"""

    # First check if the file is uploaded and email address is supplied
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Read in the file and check the number of lines.
    file = request.files["file"]
    file_data = file.read().decode("utf-8")
    num_rows = file_data.count("\n")
    print(num_rows)
    # file_data = base64.b64encode(file_data)  # Encode file as base64

    if num_rows > 201:
        return jsonify({
            "error": "Too many entries! Please reduce the number of stars to 200 or split the CSV into smaller parts."
        }), 400

    user_email = request.form.get("email", "").strip()
    if not user_email:
        return jsonify({"error": "Email is required"}), 400

    # Schedule the process
    task = upload_csv.apply_async(args=[file_data, user_email])
    return jsonify({"task_id": task.id}), 200


@shared_task(ignore_result=True)
def upload_csv(file_data, user_email):
    """Handles CSV file upload and batch computation"""
    print("I am doing something!")

    print("I REALLY AM!")
    file_data = io.BytesIO(file_data.encode())
    df = pd.read_csv(file_data)
    print(df)

    # Create a single batch directory for all results
    base_dir = "./tmp"
    os.makedirs(base_dir, exist_ok=True)
    batch_output_dir = tempfile.mkdtemp(dir=base_dir)
    os.makedirs(batch_output_dir, exist_ok=True)

    print(f"Batch directory created: {batch_output_dir}")  # Debugging

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

            results_csv_path = os.path.join(batch_output_dir, "results.csv")
            csv_header_written = False

            pdf_paths = []
            all_results = []

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

                abundances = {}
                total_metal_mass = 0
                for element, default_value in default_abundances.items():
                    if element in ["H", "HE"]:
                        abundances[element] = float(row[element]) if element in row and not pd.isna(row[element]) else default_value
                    else:
                        scaled_value = (float(row[element]) if element in row and not pd.isna(row[element]) else default_value) * zscale
                        abundances[element] = scaled_value
                        total_metal_mass += scaled_value

                zstar = calculate_metallicity_massb(abundances)

                #Keep He fixed and adjust H so the total remains 1
                helium_abundance = abundances["HE"]
                hydrogen_abundance = 1.0 - (total_metal_mass + helium_abundance)

                if hydrogen_abundance >= 0:
                    abundances["H"] = hydrogen_abundance

                # No need to make figures when running working through a csv file.
                does_plot = False

                # Start the main calculation
                results_dict = process_computation(lum, teff, mstar, zscale, zstar, helium_abundance, abundances,
                                                   pdf_name, batch_output_dir, False, does_plot)
                pdf_paths.append(pdf_path)
                all_results.append(results_dict)

            for index, row in df.iterrows():
                results_dict = all_results[index]
                pdf_name = str(row["name"])
                result_dir = os.path.join(batch_output_dir, pdf_name)
                mass_abundance_path = os.path.join(result_dir, "output", "mass_abundance")

                # Read abundances from the mass_abundance file
                abundances_data = {}
                if os.path.exists(mass_abundance_path):
                    with open(mass_abundance_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                element_symbol = " ".join(parts[1:-1]).replace("'", "").strip()
                                abundance_value = float(parts[-1])
                                abundances_data[element_symbol] = abundance_value

                with open(results_csv_path, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["Name", "Luminosity", "Teff", "Mstar", "Rstar", "log g", "Zstar", "Mass Loss Rate", "Qbar", "Alpha", "Q0", "Vinf", "Remark", *abundances_data.keys()])
                    if not csv_header_written:
                        writer.writeheader()
                        csv_header_written = True
                    if not results_dict["fail"]:
                        writer.writerow({"Name": pdf_name,
                                         "Luminosity": row["luminosity"],
                                         "Teff": row["teff"],
                                         "Mstar": row["mstar"],
                                         "Rstar": "{:.2e}".format(results_dict["R_star"]),
                                         "log g": "{:.2e}".format(results_dict["log_g"]),
                                         "Zstar": "{:.3e}".format(results_dict["Zmass"]),
                                         "Mass Loss Rate": "{:.3e}".format(results_dict["mdot"]),
                                         "Qbar": "{:.2e}".format(results_dict["Qbar"]),
                                         "Alpha": "{:.2e}".format(results_dict["alpha"]),
                                         "Q0": "{:.2e}".format(results_dict["Q0"]),
                                         "Vinf": "{:.2e}".format(results_dict["vinf"]),
                                         "Remark": results_dict["fail_reason"],
                                         **abundances_data})
                    else:
                        writer.writerow({"Name": pdf_name,
                                         "Luminosity": row["luminosity"],
                                         "Teff": row["teff"],
                                         "Mstar": row["mstar"],
                                         "Rstar": f"{np.nan}",
                                         "log g": f"{np.nan}",
                                         "Zstar": f"{np.nan}",
                                         "Mass Loss Rate": f"{np.nan}",
                                         "Qbar": f"{np.nan}",
                                         "Alpha": f"{np.nan}",
                                         "Q0": f"{np.nan}",
                                         "Vinf": f"{np.nan}",
                                         "Remark": results_dict["fail_reason"],
                                         **abundances_data})

            # Send email with ZIP
            body = load_email_body('./mailing/mail_template.j2')
            subprocess.run(["python3", "./mailing/mailer.py", "--t", user_email, "--s", "LIME Computation Results", "--b", body, "--a", results_csv_path])

        except Exception as e:
            print(f"Unexpected error in batch processing: {str(e)}")

        finally:
            # The batch directory is removed after processing
            shutil.rmtree(batch_output_dir, ignore_errors=True)

    # **Start computation in a new thread**
    batch_thread = threading.Thread(target=process_batch)
    batch_thread.start()

    return jsonify(response_message), 200
    #
    # except Exception as e:
    #     print(e)
    #     return jsonify({"error": f"Unexpected error in upload_csv: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
