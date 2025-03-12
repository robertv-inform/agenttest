# agenttest
To create conversational ai


review:
import os
import json
import pandas as pd
import numpy as np
import pickle
import traceback
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import openai

########################################
# Directory & File Setup
########################################
ROOT_DIR = os.getcwd()
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATA_DIR = os.path.join(ROOT_DIR, "data")
JOB_DESC_DIR = os.path.join(ROOT_DIR, "jobdescription")
LOG_DIR = os.path.join(ROOT_DIR, "log")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "model_cache")

for folder in [CONFIG_DIR, DATA_DIR, JOB_DESC_DIR, LOG_DIR, CACHE_DIR, MODEL_CACHE_DIR]:
    os.makedirs(folder, exist_ok=True)

# File paths
JOB_DESC_FILE = "ms_Director__AppliedScience_jd1.txt"
JOB_DESC_BASENAME = os.path.splitext(JOB_DESC_FILE)[0]  # e.g., ms_Director__AppliedScience_jd1
KEY_ATTR_FILE = os.path.join(CONFIG_DIR, f"{JOB_DESC_BASENAME}_kab.json")
WEIGHT_CONFIG_FILE = os.path.join(CONFIG_DIR, "keyattribute_weights.json")
RESUME_CSV_FILE = os.path.join(DATA_DIR, "final_candidate_resumes_complete.csv")
OUTPUT_LOG = os.path.join(LOG_DIR, "ranked_candidates_with_explanation.csv")

# Set your OpenAI API key (if using real GPT-4o; currently the call is mocked)
openai.api_key = "YOUR_OPENAI_API_KEY"

########################################
# Option: Choose caching method
########################################
USE_FAISS = True  # Set this to True to enable FAISS caching

########################################
# FAISS Caching Functions (if enabled)
########################################
if USE_FAISS:
    import faiss
    DIMENSION = 384  # This should match the embedding dimension of your model

    FAISS_INDEX_FILE = os.path.join(CACHE_DIR, "faiss_embedding_cache.pkl")
    
    def load_faiss_index():
        if os.path.exists(FAISS_INDEX_FILE):
            with open(FAISS_INDEX_FILE, "rb") as f:
                data = pickle.load(f)
                return data["index"], data["texts"]
        else:
            index = faiss.IndexFlatL2(DIMENSION)
            return index, []

    def save_faiss_index(index, texts):
        with open(FAISS_INDEX_FILE, "wb") as f:
            pickle.dump({"index": index, "texts": texts}, f)

    def faiss_embed_text(text, model, index, text_cache):
        if text in text_cache:
            idx = text_cache.index(text)
            return index.reconstruct(idx)
        emb = model.encode(text).astype(np.float32)
        index.add(np.array([emb]))
        text_cache.append(text)
        save_faiss_index(index, text_cache)
        return emb

########################################
# 1) GPT-4o Key-Attribute Extraction
########################################
def call_gpt4o_for_attributes(job_desc_text):
    """
    Call OpenAI's GPT-4o using a two-shot prompt style to extract key attributes.
    The prompt is written using an f""" """ multi-line string.
    If the call succeeds, the extracted JSON is returned.
    In case of any error, None is returned.
    """
    try:
        prompt = f"""
SYSTEM INSTRUCTIONS:

Translation to English:
If the job description is in multiple languages or not in English, translate the entire text into English first.

Strict Extraction Rules:
Extract and normalize all relevant job attributes without making assumptions.
Capture experience requirements correctly. For experience, return a field "experience" that includes:
  - "total_experience": the overall minimum years required, formatted as {{"gte": X}} (e.g., {{"gte": 8}}),
  - For each degree level (e.g., "Bachelor's", "Master's", "Doctorate"), return the minimum years required, formatted as {{"gte": X}}.
For education, return an array where each object contains:
  - "degree": the required degree level (e.g., "Bachelor's", "Master's", "Doctorate"),
  - "abbreviation": the common short form (e.g., "BSc", "MSc", "PhD"),
  - "fields": a list of relevant fields; if not specified, use ["Computer Science", "Econometrics"].
For skills, certifications, and languages, return lists of the required items.
Strict JSON Formatting Guidelines:
- Return only the JSON response—do not include explanations or extra text.
- The JSON must be strictly formatted with an indentation of 4 spaces.
- If a field is not available, return it as an empty string ("") or an empty list ([]).
- Ensure the output is strictly parseable as JSON.

Process the Following Job Description:
{job_desc_text}

JSON Output Format:
{{
    "location": ["List of locations mentioned in the job description, specific city or region if applicable"],
    "experience": {{
         "total_experience": {{"gte": 8}},
         "Bachelor's": {{"gte": 8}},
         "Master's": {{"gte": 6}},
         "Doctorate": {{"gte": 5}}
    }},
    "skills": ["List of all technical, domain-specific, hard, and soft skills"],
    "education": [
        {{
            "degree": "Bachelor's",
            "abbreviation": "BSc",
            "fields": ["Computer Science", "Econometrics"]
        }},
        {{
            "degree": "Master's",
            "abbreviation": "MSc",
            "fields": ["Computer Science", "Econometrics"]
        }},
        {{
            "degree": "Doctorate",
            "abbreviation": "PhD",
            "fields": ["Computer Science", "Econometrics"]
        }}
    ],
    "certifications": ["List of certifications if any"],
    "languages": ["List of languages required for the role"]
}}

Final Instructions for GPT-4o:
Extract all job attributes from the provided job description exactly as per the format above. Do not include any extra commentary.
        """
        # Uncomment these lines to use a real GPT-4o API call:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "system", "content": prompt}],
        #     temperature=0
        # )
        # extracted_text = response["choices"][0]["message"]["content"]
        # extracted = json.loads(extracted_text)
        # return extracted

        # For demonstration, return a mock response:
        mock_response_json = {
            "location": ["Multiple Locations, United States"],
            "experience": {
                "total_experience": {"gte": 8},
                "Bachelor's": {"gte": 8},
                "Master's": {"gte": 6},
                "Doctorate": {"gte": 5}
            },
            "skills": [
                "AI",
                "Machine Learning",
                "Natural Language Processing",
                "Management",
                "Data Science"
            ],
            "education": [
                {
                    "degree": "Bachelor's",
                    "abbreviation": "BSc",
                    "fields": ["Computer Science", "Econometrics"]
                },
                {
                    "degree": "Master's",
                    "abbreviation": "MSc",
                    "fields": ["Computer Science", "Econometrics"]
                },
                {
                    "degree": "Doctorate",
                    "abbreviation": "PhD",
                    "fields": ["Computer Science", "Econometrics"]
                }
            ],
            "certifications": [],
            "languages": ["English"]
        }
        return mock_response_json

    except Exception as e:
        print(f"[call_gpt4o_for_attributes] GPT-4o call failed: {e}")
        return None

def get_or_create_key_attributes():
    """
    Read the job description from the jobdescription folder,
    call GPT-4o to extract key attributes using the above function, and
    overwrite the dynamic key attribute builder file with the new response.
    If the API call fails or an error occurs, fall back to a default empty key attribute JSON.
    """
    jd_path = os.path.join(JOB_DESC_DIR, JOB_DESC_FILE)
    if not os.path.exists(jd_path):
        print(f"Job description file '{JOB_DESC_FILE}' not found in {JOB_DESC_DIR}. Using default empty key attributes.")
        return {
            "location": [],
            "experience": {"total_experience": {"gte": 0}, "Bachelor's": {"gte": 0}, "Master's": {"gte": 0}, "Doctorate": {"gte": 0}},
            "skills": [],
            "education": [],
            "certifications": [],
            "languages": []
        }
    
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()
    
    extracted = call_gpt4o_for_attributes(jd_text)
    if extracted is not None:
        try:
            with open(KEY_ATTR_FILE, "w", encoding="utf-8") as fw:
                json.dump(extracted, fw, indent=4)
            print(f"GPT-4o extraction successful. Updated {KEY_ATTR_FILE}.")
            return extracted
        except Exception as e:
            print(f"Error writing to {KEY_ATTR_FILE}: {e}. Using default empty key attributes.")
            return {
                "location": [],
                "experience": {"total_experience": {"gte": 0}, "Bachelor's": {"gte": 0}, "Master's": {"gte": 0}, "Doctorate": {"gte": 0}},
                "skills": [],
                "education": [],
                "certifications": [],
                "languages": []
            }
    else:
        print("GPT-4o extraction failed. Using default empty key attributes.")
        return {
            "location": [],
            "experience": {"total_experience": {"gte": 0}, "Bachelor's": {"gte": 0}, "Master's": {"gte": 0}, "Doctorate": {"gte": 0}},
            "skills": [],
            "education": [],
            "certifications": [],
            "languages": []
        }

########################################
# 2) Resume Data Parsing
########################################
def parse_resume_data():
    """
    Parse the resume CSV file and extract candidate details.
    Expects a CSV with a 'candidate_id' column and a 'Resume' column containing JSON text.
    """
    if not os.path.exists(RESUME_CSV_FILE):
        raise FileNotFoundError(f"Resume CSV not found at: {RESUME_CSV_FILE}")
    
    df = pd.read_csv(RESUME_CSV_FILE)
    candidates = []
    
    for _, row in df.iterrows():
        candidate_id = row.get("candidate_id", f"cand_{_}")
        resume_str = row.get("Resume", "{}")
        try:
            resume_json = json.loads(resume_str)
        except Exception:
            resume_json = {}
        
        # For location: extract multiple subfields from "address"
        address = resume_json.get("personal", {}).get("address", {})
        candidate_locations = []
        if address.get("city"):
            candidate_locations.append(address.get("city").strip())
        if address.get("subRegion", {}).get("description"):
            candidate_locations.append(address.get("subRegion", {}).get("description").strip())
        if address.get("region", {}).get("description"):
            candidate_locations.append(address.get("region", {}).get("description").strip())
        if address.get("country", {}).get("description"):
            candidate_locations.append(address.get("country", {}).get("description").strip())
        
        # Extract total experience as a numeric value.
        total_exp = float(resume_json.get("summary", {}).get("totalExperienceYears", 0))
        
        # For skills: robust extraction handling various types.
        skills_data = resume_json.get("skills", [])
        skills_list = []
        if isinstance(skills_data, dict):
            for key, val in skills_data.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            sdesc = item.get("skillDescription", "")
                            if sdesc:
                                skills_list.append(sdesc.lower())
                        elif isinstance(item, str):
                            skills_list.append(item.lower())
        elif isinstance(skills_data, list):
            for item in skills_data:
                if isinstance(item, dict):
                    sdesc = item.get("skillDescription", "")
                    if sdesc:
                        skills_list.append(sdesc.lower())
                elif isinstance(item, str):
                    skills_list.append(item.lower())
        elif isinstance(skills_data, str):
            skills_list = [x.strip().lower() for x in skills_data.split(",")]
        
        # For education: try "localDescription" first, then "degreeName"
        education_list = []
        for deg in resume_json.get("educationHistory", {}).get("degrees", []):
            edu_val = deg.get("localDescription", "").strip()
            if not edu_val:
                edu_val = deg.get("degreeName", "").strip()
            if edu_val:
                education_list.append(edu_val.lower())
        
        # For certifications:
        certifications_data = resume_json.get("certifications", [])
        certifications_list = []
        if isinstance(certifications_data, list):
            for cert in certifications_data:
                if isinstance(cert, dict):
                    cname = cert.get("certificationName", "")
                    if cname:
                        certifications_list.append(cname.lower())
                elif isinstance(cert, str):
                    certifications_list.append(cert.lower())
        elif isinstance(certifications_data, str):
            certifications_list = [x.strip().lower() for x in certifications_data.split(",")]
        
        # For languages: check "languageSkills", fallback to top-level "lang"
        languages_data = resume_json.get("languageSkills", [])
        languages_list = []
        if isinstance(languages_data, list):
            for lang in languages_data:
                if isinstance(lang, dict):
                    ldesc = lang.get("language", "")
                    if ldesc:
                        languages_list.append(ldesc.lower())
                elif isinstance(lang, str):
                    languages_list.append(lang.lower())
        elif isinstance(languages_data, str):
            languages_list = [x.strip().lower() for x in languages_data.split(",")]
        if not languages_list and resume_json.get("lang"):
            languages_list.append(resume_json.get("lang").lower())
        
        candidate_dict = {
            "candidate_id": str(candidate_id),
            "location": candidate_locations,
            "skills": skills_list,
            "education": education_list,
            "certifications": certifications_list,
            "languages": languages_list,
            "total_experience": total_exp  # numeric value for experience comparison
        }
        candidates.append(candidate_dict)
    
    return candidates

########################################
# 3) Embedding Model & Caching Options
########################################
def get_model():
    """
    Load the SentenceTransformer model. If not available locally in MODEL_CACHE_DIR,
    download and save it.
    """
    model_path = os.path.join(MODEL_CACHE_DIR, "all-MiniLM-L6-v2")
    if not os.path.exists(model_path):
        print("Model not found locally. Downloading all-MiniLM-L6-v2 to model_cache folder...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(model_path)
    else:
        model = SentenceTransformer(model_path)
    return model

embedding_cache = {}

def simple_embed_text(text, model, cache):
    """
    Compute and cache embeddings using a simple in-memory dictionary.
    """
    if text in cache:
        return cache[text]
    emb = model.encode(text).astype(np.float32)
    cache[text] = emb
    return emb

def get_embedding(text, model, faiss_index=None, faiss_cache=None):
    """
    Wrapper to get embedding using either FAISS caching or simple in-memory caching.
    """
    if USE_FAISS:
        return faiss_embed_text(text, model, faiss_index, faiss_cache)
    else:
        return simple_embed_text(text, model, embedding_cache)

########################################
# 4) Similarity & Weighted Scoring Functions
########################################
def multi_value_similarity(jd_list, cand_list, model, faiss_index=None, faiss_cache=None):
    """
    For attributes with multiple values (e.g., skills, certifications, location):
    For each value in the job description list, find the candidate value with the highest cosine similarity.
    Returns the average similarity and an explanation string.
    """
    if not jd_list or not cand_list:
        return 0.0, ""
    
    total_sim = 0.0
    explanation = ""
    count = 0
    for jd_item in jd_list:
        jd_emb = get_embedding(jd_item, model, faiss_index, faiss_cache)
        best_score = 0.0
        best_candidate = ""
        for cand_item in cand_list:
            cand_emb = get_embedding(cand_item, model, faiss_index, faiss_cache)
            sim = util.pytorch_cos_sim(jd_emb, cand_emb).item()
            if sim > best_score:
                best_score = sim
                best_candidate = cand_item
        total_sim += best_score
        explanation += f"Matched '{jd_item}' with '{best_candidate}' (sim: {best_score:.2f}); "
        count += 1
    average_sim = total_sim / count if count > 0 else 0.0
    return average_sim, explanation

def single_value_similarity(jd_val, cand_val, model, faiss_index=None, faiss_cache=None):
    """
    Compute similarity for single string values.
    """
    if not jd_val or not cand_val:
        return 0.0, ""
    
    if isinstance(jd_val, list):
        jd_val = " ".join(jd_val)
    if isinstance(cand_val, list):
        cand_val = " ".join(cand_val)
    
    jd_emb = get_embedding(jd_val, model, faiss_index, faiss_cache)
    cand_emb = get_embedding(cand_val, model, faiss_index, faiss_cache)
    sim = util.pytorch_cos_sim(jd_emb, cand_emb).item()
    explanation = f"Compared '{jd_val}' with '{cand_val}' (sim: {sim:.2f}); "
    return sim, explanation

def load_weights():
    """
    Load and normalize the weight configuration from the config folder.
    Weights provided (e.g., on a 0-100 scale) are normalized to sum to 1.
    """
    if not os.path.exists(WEIGHT_CONFIG_FILE):
        raise FileNotFoundError(f"No weight config found at {WEIGHT_CONFIG_FILE}")
    with open(WEIGHT_CONFIG_FILE, "r", encoding="utf-8") as f:
        weight_data = json.load(f)
    total = sum(weight_data.values())
    for key in weight_data:
        weight_data[key] = weight_data[key] / total
    return weight_data

def compute_candidate_score(jd_attrs, cand_attrs, weights, model, faiss_index=None, faiss_cache=None):
    """
    Compute the candidate score by comparing each attribute.
    Special handling:
      - For "experience": compare candidate's total experience against required "total_experience".
      - For "education": use OR logic. If candidate meets any one required degree's experience threshold,
        then education match = 1; otherwise, 0.
    For other attributes, use embedding-based similarity.
    The final score is the weighted sum of each attribute's match score.
    """
    total_score = 0.0
    explanation = ""
    
    # EXPERIENCE: Compare candidate's total experience against required "total_experience"
    if isinstance(jd_attrs.get("experience", {}), dict):
        required_total = jd_attrs["experience"].get("total_experience", {}).get("gte", 0)
        candidate_exp = cand_attrs.get("total_experience", 0)
        exp_sim = 1.0 if candidate_exp >= required_total else 0.0
        explanation += f"Total Experience: required gte {required_total}, candidate {candidate_exp} -> sim: {exp_sim}; "
        total_score += exp_sim * weights.get("experience", 0)
    else:
        sim, exp = multi_value_similarity(jd_attrs.get("experience", []), cand_attrs.get("experience", []), model, faiss_index, faiss_cache)
        total_score += sim * weights.get("experience", 0)
        explanation += f"Experience -> {exp} weighted sim: {sim * weights.get('experience', 0):.2f}; "
    
    # EDUCATION: Use OR logic. If candidate meets at least one required degree's experience threshold, education score = 1.
    edu_reqs = jd_attrs.get("education", [])
    education_sim = 0.0
    edu_explanation = ""
    candidate_edu = [edu.lower() for edu in cand_attrs.get("education", [])]
    candidate_exp = cand_attrs.get("total_experience", 0)
    for req in edu_reqs:
        req_degree = req.get("degree", "").lower()
        req_exp = req.get("experience", {}).get("gte", 0)
        if any(req_degree in edu for edu in candidate_edu):
            if candidate_exp >= req_exp:
                education_sim = 1.0
                edu_explanation += f"Matched {req_degree} (required gte {req_exp}, candidate exp {candidate_exp}); "
                break
            else:
                edu_explanation += f"{req_degree} found but candidate exp {candidate_exp} < required {req_exp}; "
        else:
            edu_explanation += f"{req_degree} not found; "
    total_score += education_sim * weights.get("education", 0)
    explanation += f"Education -> {edu_explanation} weighted sim: {education_sim * weights.get('education', 0):.2f}; "
    
    # Other Attributes: location, skills, certifications, languages
    for attr in ["location", "skills", "certifications", "languages"]:
        jd_value = jd_attrs.get(attr, [])
        cand_value = cand_attrs.get(attr, [])
        sim, exp = multi_value_similarity(jd_value, cand_value, model, faiss_index, faiss_cache)
        weighted_sim = sim * weights.get(attr, 0)
        total_score += weighted_sim
        explanation += f"{attr.capitalize()} -> {exp} weighted sim: {weighted_sim:.2f}; "
    
    return total_score, explanation

########################################
# 5) Main Execution: Ranking Candidates
########################################
def main():
    try:
        print("Extracting key attributes from the job description...")
        jd_key_attrs = get_or_create_key_attributes()
        print("Job Description Key Attributes:")
        print(json.dumps(jd_key_attrs, indent=4))
        
        print("\nParsing candidate resumes...")
        candidates = parse_resume_data()
        print(f"Found {len(candidates)} candidates.")
        
        print("\nLoading weight configuration...")
        weights = load_weights()
        
        print("Loading SentenceTransformer model...")
        model = get_model()
        
        if USE_FAISS:
            from faiss import IndexFlatL2
            DIMENSION = 384
            if os.path.exists(FAISS_INDEX_FILE):
                with open(FAISS_INDEX_FILE, "rb") as f:
                    data = pickle.load(f)
                    faiss_index = data["index"]
                    faiss_cache = data["texts"]
            else:
                faiss_index = IndexFlatL2(DIMENSION)
                faiss_cache = []
        else:
            faiss_index = None
            faiss_cache = None
        
        results = []
        for cand in candidates:
            score, exp = compute_candidate_score(jd_key_attrs, cand, weights, model, faiss_index, faiss_cache)
            cand["score"] = score
            cand["explanation"] = exp
            results.append(cand)
        
        df = pd.DataFrame(results)
        df = df.sort_values(by="score", ascending=False)
        print("\nRanked Candidates:")
        print(df[["candidate_id", "score", "explanation"]])
        
        df.to_csv(OUTPUT_LOG, index=False)
        print(f"\nResults saved to {OUTPUT_LOG}")
        print("\nNote: In-memory caching is used by default. Set USE_FAISS = True to enable FAISS caching.")
    
    except Exception as e:
        print(f"Error in candidate evaluation system: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

