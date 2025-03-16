import os
import json
import glob
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from candidate_evaluation_system import ConfigManager, KeyAttributeExtractor, CandidateScoreService

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

config = ConfigManager()
# Use a separate CSV for JSON resume rankings.
JSON_RANKING_FILE = os.path.join(config.LOG_DIR, "json_candidate_ranking.csv")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_job', methods=['POST'])
def upload_job():
    if 'job_description' not in request.files:
        return "No job description file uploaded.", 400
    file = request.files['job_description']
    if file.filename == '':
        return "No file selected.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    with open(filepath, "r", encoding="utf-8") as f:
        job_text = f.read()
    
    extractor = KeyAttributeExtractor(config)
    kab_json = extractor.call_gpt4o_for_attributes(job_text)
    
    base = os.path.splitext(filename)[0]
    kab_filename = f"{base}_kab.json"
    kab_filepath = os.path.join(config.CONFIG_DIR, kab_filename)
    with open(kab_filepath, "w", encoding="utf-8") as f:
        json.dump(kab_json, f, indent=4)
        
    return "Job description processed and KAB saved.", 200

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    # Get the latest *_kab.json file from the config folder.
    kab_files = glob.glob(os.path.join(config.CONFIG_DIR, "*_kab.json"))
    if not kab_files:
        return "No job description KAB available. Please upload a job description first.", 400
    kab_filepath = max(kab_files, key=os.path.getmtime)
    try:
        with open(kab_filepath, "r", encoding="utf-8") as f:
            kab_json = json.load(f)
    except Exception as e:
        return f"Error loading KAB JSON: {str(e)}", 500

    if 'resume_json' not in request.files:
        return "No resume JSON file uploaded.", 400
    file = request.files['resume_json']
    if file.filename == '':
        return "No resume file selected.", 400

    resume_data = file.read().decode("utf-8")
    try:
        resume_json = json.loads(resume_data)
    except Exception as e:
        return f"Invalid JSON format: {str(e)}", 400

    # Filter resume JSON to keep only expected keys.
    expected_keys = {"lang", "personal", "summary", "educationHistory", "employmentHistory", "skills"}
    filtered_resume = {k: v for k, v in resume_json.items() if k in expected_keys}
    
    service = CandidateScoreService()
    score, explanation = service.score_candidate(kab_json, filtered_resume)
    
    candidate_id = filtered_resume.get("personal", {}).get("completeName", "unknown")
    candidate_result = {
        "candidate_id": candidate_id,
        "score": score,
        "explanation": explanation,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if os.path.exists(JSON_RANKING_FILE):
        df = pd.read_csv(JSON_RANKING_FILE)
    else:
        df = pd.DataFrame(columns=["candidate_id", "score", "explanation", "timestamp"])
    df = pd.concat([df, pd.DataFrame([candidate_result])], ignore_index=True)
    df.to_csv(JSON_RANKING_FILE, index=False)
    
    return "Resume processed and candidate score computed.", 200

@app.route('/scoring_results', methods=['GET'])
def scoring_results():
    if os.path.exists(JSON_RANKING_FILE):
        df = pd.read_csv(JSON_RANKING_FILE)
        df = df.sort_values(by="score", ascending=False)
        # Use escape=False so that HTML in the explanation column is rendered as HTML.
        table_html = df.to_html(classes="table table-striped", index=False, escape=False)
        return render_template('results.html', table_html=table_html)
    else:
        return "No candidate scoring results available.", 200

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
