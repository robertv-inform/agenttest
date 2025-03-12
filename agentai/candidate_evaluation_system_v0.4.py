import os
import json
import pandas as pd
import numpy as np
import faiss
import pickle
import traceback
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import openai

########################################
# Directory Setup
########################################
ROOT_DIR = os.getcwd()  # Or wherever your project root is
LOG_DIR = os.path.join(ROOT_DIR, "log")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATA_DIR = os.path.join(ROOT_DIR, "data")
JOB_DESC_DIR = os.path.join(ROOT_DIR, "jobdescription")
MODEL_REPO = os.path.join(ROOT_DIR, "model_repo")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(JOB_DESC_DIR, exist_ok=True)
os.makedirs(MODEL_REPO, exist_ok=True)

# Files
JOB_DESC_FILE = "ms_Director__AppliedScience_jd1.txt"
KEY_ATTR_FALLBACK = os.path.join(CONFIG_DIR, "key_attribute_builder.json")  # fallback
WEIGHT_CONFIG_FILE = os.path.join(CONFIG_DIR, "keyattribute_weights.json")
RESUME_CSV_FILE = os.path.join(DATA_DIR, "final_candidate_resumes_complete.csv")
OUTPUT_LOG = os.path.join(LOG_DIR, "ranked_candidates_with_explanation.csv")

# GPT-4o related
openai.api_key = "YOUR_OPENAI_API_KEY"  # Put your real API key or handle via env vars

########################################
# 1) GPT-4o Key-Attribute Extraction
########################################
def call_gpt4o_for_attributes(job_desc_text):
    """
    Attempts a GPT-4o call with a 2-shot style approach to extract key attributes.
    If it fails, return None.
    """

    try:
        # Updated system instructions with your new prompt
        system_instructions = (
            "SYSTEM INSTRUCTIONS:\n\n"
            "Translation to English\n"
            "If the job description is in multiple languages or not in English, translate the entire text into English first.\n"
            "Strict Extraction Rules\n"
            "Extract and normalize all relevant job attributes without making assumptions.\n"
            "Ensure extracted information strictly follows the JSON format provided below.\n"
            "Capture experience requirements correctly, ensuring education is extracted in an internationally standardized format (e.g., Ph.D., Master's, Bachelor's, Diploma).\n"
            "Capture all nuances in education details, including abbreviations, fields of study, specific qualifications, and additional educational information.\n"
            "Strict JSON Formatting Guidelines\n"
            "- Return only the JSON response—do not include explanations, commentary, or extra text.\n"
            "- The JSON must be strictly formatted with an indentation of 4 spaces.\n"
            "- If a field is not available in the job description, return it as an empty string (\"\") or an empty list ([]).\n"
            "- Ensure JSON validity—the output must be strictly parseable with no missing commas, trailing characters, or formatting errors.\n\n"
            "Process the Following Job Description\n\n"
            "{job_description}\n\n"
            "JSON Output Format\n"
            "Return the extracted information strictly in the following JSON structure:\n\n"
            "{\n"
            "    \"location\": [\"List of locations mentioned in the job description, Specific city or region if applicable\"],\n"
            "    \"experience\": [\n"
            "        {\n"
            "            \"years\": \"Number of years required\",\n"
            "            \"education_level\": \"Ph.D. | Master's | Bachelor's | Diploma (International standard format)\",\n"
            "            \"skills\": [\"List of specific skills required for that experience level\"]\n"
            "        }\n"
            "    ],\n"
            "    \"skills\": [\"List of all technical, domain-specific, hard, and soft skills\"],\n"
            "    \"education\": [\n"
            "        {\n"
            "            \"degree\": \"Ph.D. | Master's | Bachelor's | Diploma\",\n"
            "            \"abbreviation\": \"Short form if applicable (e.g., B.Sc., M.Tech.)\",\n"
            "            \"fields\": [\"Relevant field of study (e.g., Computer Science, Finance)\"],\n"
            "            \"specific_qualifications\": [\"Any additional qualifications if specified\"],\n"
            "            \"other_education_related_info\": \"Extra details related to education\"\n"
            "        }\n"
            "    ],\n"
            "    \"certifications\": [\n"
            "        {\n"
            "            \"name\": \"Full name of the certification (e.g., Project Management Professional)\",\n"
            "            \"abbreviation\": \"Short form (e.g., PMP) if applicable\"\n"
            "        }\n"
            "    ],\n"
            "    \"languages\": [\"List of languages required for the role\"]\n"
            "}\n\n"
            "Handling Experience & Education (Strict JSON Compliance)\n"
            "- If multiple experience levels are given based on education, capture each separately.\n"
            "- Use international standard education formats (Ph.D., Master's, Bachelor's, Diploma).\n"
            "- Include only skills specific to that experience level (avoid redundant listing of education in \"skills\").\n"
            "- Capture all additional education-related details, including abbreviations, relevant fields, specific qualifications, and extra notes.\n"
            "- No additional or unnecessary keys should be included.\n\n"
            "Final Instructions for GPT-4o\n"
            "Extract all job attributes from the provided job description below.\n\n"
            "Ensure JSON is strictly formatted with correct indentation (4 spaces) and contains no extra text.\n"
            "Do not add any commentary or extra explanations—only return valid JSON.\n"
            "Ensure \"experience\" is correctly mapped to education levels, listing each separately.\n"
            "Use international education standards (Ph.D., Master's, Bachelor's, Diploma).\n"
            "Only capture skills relevant to that experience level in \"experience\".\n"
            "Ensure \"education\" captures all nuances, including abbreviations, relevant fields, specific qualifications, and additional educational information.\n"
            "If a field is not found, return an empty string (\"\") or an empty list ([]).\n"
            "Return only the JSON output in the exact format provided. Do not include explanations, additional text, or incorrect formatting.\n"
        ).format(job_description=job_desc_text)

        # Mock the response as if from GPT-4o (no actual API call here).
        mock_response_json = {
            "location": ["Multiple Locations, United States"],
            "experience": [
                {
                    "years": "8",
                    "education_level": "Bachelor's",
                    "skills": ["Machine Learning", "NLP"]
                }
            ],
            "skills": ["AI", "Machine Learning", "Natural Language Processing"],
            "education": [
                {
                    "degree": "Bachelor's",
                    "abbreviation": "B.Sc.",
                    "fields": ["Computer Science"],
                    "specific_qualifications": [],
                    "other_education_related_info": ""
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
    Reads the job description from 'jobdescription/ms_Director__AppliedScience_jd1.txt',
    calls GPT-4o, and if it fails, falls back to the existing local key_attribute_builder.json.
    If GPT-4o succeeds, overwrites the local JSON.
    """
    jd_path = os.path.join(JOB_DESC_DIR, JOB_DESC_FILE)
    if not os.path.exists(jd_path):
        print(f"Job description file {JOB_DESC_FILE} not found in jobdescription folder. Using fallback JSON.")
        if os.path.exists(KEY_ATTR_FALLBACK):
            with open(KEY_ATTR_FALLBACK, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise FileNotFoundError("No fallback key_attribute_builder.json found either!")

    # Read the job description text
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # Attempt GPT-4o extraction
    extracted = call_gpt4o_for_attributes(jd_text)
    if extracted is not None:
        # Overwrite local fallback
        with open(KEY_ATTR_FALLBACK, "w", encoding="utf-8") as fw:
            json.dump(extracted, fw, indent=4)
        return extracted
    else:
        # Fallback
        print("GPT-4o extraction failed. Using existing key_attribute_builder.json.")
        if os.path.exists(KEY_ATTR_FALLBACK):
            with open(KEY_ATTR_FALLBACK, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            raise FileNotFoundError("No fallback key_attribute_builder.json found!")

########################################
# 2) Resume Parsing
########################################
def parse_resume_data():
    """
    final_candidate_resumes_complete.csv is expected to have columns like:
      candidate_id, Resume (which is a JSON text),
      or other columns if you prefer. We'll parse them to extract location, experience, skills, etc.
    """
    if not os.path.exists(RESUME_CSV_FILE):
        raise FileNotFoundError(f"Resume CSV not found at: {RESUME_CSV_FILE}")

    df = pd.read_csv(RESUME_CSV_FILE)
    # We'll create a list of candidate dicts
    candidates = []

    for _, row in df.iterrows():
        cand_id = row.get("candidate_id", f"cand_{_}")
        resume_str = row.get("Resume", "{}")

        try:
            resume_json = json.loads(resume_str)
        except:
            resume_json = {}

        # Extract location
        location = ""
        address = resume_json.get("personal", {}).get("address", {})
        city = address.get("city", "")
        country = address.get("country", "")
        location = f"{city} {country}".strip()

        # Extract total_experience (float)
        summary = resume_json.get("summary", {})
        total_exp = float(summary.get("totalExperienceYears", 0))

        # Extract skills from 'skills' section
        skill_list = []
        if "skills" in resume_json:
            # The resume might have a "skills" list of objects
            for skill_obj in resume_json["skills"]:
                sdesc = skill_obj.get("skillDescription", "")
                if sdesc:
                    skill_list.append(sdesc.lower())

        # Extract education info
        education_list = []
        edu_hist = resume_json.get("educationHistory", {}).get("degrees", [])
        for deg in edu_hist:
            dname = deg.get("degreeName", "")
            education_list.append(dname.lower())

        # Extract certifications
        cert_list = []
        cert_data = resume_json.get("certifications", [])
        for c in cert_data:
            cert_name = c.get("certificationName", "")
            cert_list.append(cert_name)

        # Extract languages
        lang_list = []
        lang_data = resume_json.get("languageSkills", [])
        for l in lang_data:
            lang_desc = l.get("language", "")
            lang_list.append(lang_desc)

        # Build a structured dictionary
        candidate_dict = {
            "candidate_id": str(cand_id),
            "location": [location] if location else [],
            "experience": [f"{total_exp} years"] if total_exp else [],
            "skills": skill_list,
            "education": education_list,
            "certifications": cert_list,
            "languages": lang_list
        }
        candidates.append(candidate_dict)

    return candidates

########################################
# 3) Model & FAISS Embeddings
########################################
def get_model():
    """
    Load 'all-MiniLM-L6-v2' from 'model_repo' if present; otherwise download.
    """
    model_path = os.path.join(MODEL_REPO, "all-MiniLM-L6-v2")
    if not os.path.exists(model_path):
        print("Model not found locally. Downloading all-MiniLM-L6-v2 to model_repo...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        model.save(model_path)
    else:
        model = SentenceTransformer(model_path)
    return model

DIMENSION = 384
FAISS_INDEX_FILE = os.path.join(LOG_DIR, "faiss_embedding_cache.pkl")  # storing in log folder

def load_faiss_index():
    """
    Load or create a FAISS index, used for caching text embeddings.
    """
    if os.path.exists(FAISS_INDEX_FILE):
        with open(FAISS_INDEX_FILE, "rb") as f:
            data = pickle.load(f)
            index = data["index"]
            texts = data["texts"]
            return index, texts
    else:
        index = faiss.IndexFlatL2(DIMENSION)
        return index, []

def save_faiss_index(index, texts):
    with open(FAISS_INDEX_FILE, "wb") as f:
        pickle.dump({"index": index, "texts": texts}, f)

########################################
# 4) Weighted Scoring
########################################
def load_weights():
    """
    Reads weight config (0–100). We'll convert them to fractions (0–1).
    """
    if not os.path.exists(WEIGHT_CONFIG_FILE):
        raise FileNotFoundError(f"No weight config found at {WEIGHT_CONFIG_FILE}")
    with open(WEIGHT_CONFIG_FILE, "r", encoding="utf-8") as f:
        weight_data = json.load(f)
    sum_weights = sum(weight_data.values())
    for k, v in weight_data.items():
        weight_data[k] = v / sum_weights
    return weight_data

def embed_text(text, model, index, text_cache):
    """
    Use FAISS to store or retrieve embeddings for 'text'.
    text_cache is a list of already-indexed strings.
    index is the FAISS index.
    """
    if text in text_cache:
        idx = text_cache.index(text)
        return index.reconstruct(idx)

    emb = model.encode(text).astype(np.float32)
    index.add(np.array([emb]))
    text_cache.append(text)

    save_faiss_index(index, text_cache)
    return emb

def multi_value_similarity(jd_list, cand_list, model, index, text_cache):
    """
    For each item in jd_list, find best match in cand_list and average.
    """
    if not jd_list or not cand_list:
        return 0.0
    total_sim = 0.0
    count = 0
    for jd_item in jd_list:
        jd_emb = embed_text(jd_item, model, index, text_cache)
        best_score = 0.0
        for citem in cand_list:
            c_emb = embed_text(citem, model, index, text_cache)
            sim = util.pytorch_cos_sim(jd_emb, c_emb).item()
            if sim > best_score:
                best_score = sim
        total_sim += best_score
        count += 1
    return total_sim / count if count > 0 else 0.0

def single_value_similarity(jd_val, cand_val, model, index, text_cache):
    """
    For single-value fields, treat them as single strings.
    """
    if not jd_val or not cand_val:
        return 0.0

    if isinstance(jd_val, list):
        jd_val = " ".join(jd_val)
    if isinstance(cand_val, list):
        cand_val = " ".join(cand_val)

    jd_emb = embed_text(jd_val, model, index, text_cache)
    c_emb = embed_text(cand_val, model, index, text_cache)
    return util.pytorch_cos_sim(jd_emb, c_emb).item()

def compute_candidate_score(jd_attrs, candidate_attrs, weights, model, index, text_cache):
    """
    For each attribute, compute similarity, multiply by weight fraction, sum total.
    Return (score, explanation).

    Algorithm: Weighted Summation of attribute similarities.
    Each attribute subscore is 0.0 - 1.0, then multiplied by weight fraction.
    If the JD doesn't list an attribute or the candidate doesn't have it, the subscore is 0.
    """
    total_score = 0.0
    explanation = {}
    for attr, wfrac in weights.items():
        jd_val = jd_attrs.get(attr, [])
        cand_val = candidate_attrs.get(attr, [])

        if attr in ["skills", "certifications", "languages"]:
            sim = multi_value_similarity(jd_val, cand_val, model, index, text_cache)
        else:
            sim = single_value_similarity(jd_val, cand_val, model, index, text_cache)

        contribution = sim * wfrac
        total_score += contribution

        explanation[attr] = {
            "jd_value": jd_val,
            "candidate_value": cand_val,
            "similarity": round(sim, 4),
            "weight_fraction": round(wfrac, 4),
            "weighted_contribution": round(contribution, 4)
        }

    return round(total_score, 4), explanation

########################################
# 5) Main Program
########################################
def main():
    # 1) Attempt to get JD attributes from GPT-4o or fallback
    jd_attrs = get_or_create_key_attributes()

    # 2) Parse candidate resumes
    candidates = parse_resume_data()

    # 3) Load model from model_repo (download if missing)
    st_model = get_model()

    # 4) Prepare FAISS index
    faiss_index, text_cache = load_faiss_index()

    # 5) Load weight config
    weights = load_weights()

    # 6) For each candidate, compute score + explanation
    results = []
    for cand in candidates:
        score, expl = compute_candidate_score(
            jd_attrs, cand, weights,
            st_model, faiss_index, text_cache
        )
        results.append({
            "candidate_id": cand["candidate_id"],
            "score": score,
            "explanation": expl
        })

    # 7) Sort by descending score
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    # 8) Convert to DataFrame, store in log folder
    df_out = pd.DataFrame(results_sorted)
    df_out.to_csv(OUTPUT_LOG, index=False)

    print(f"\nCompleted ranking. Results saved to {OUTPUT_LOG}")
    print(df_out[["candidate_id", "score"]])  # quick view

    # Optional: also print a quick breakdown
    print("\nSample Explanation for top candidate:\n")
    if len(results_sorted) > 0:
        top_cand = results_sorted[0]
        tid = top_cand["candidate_id"]
        tscore = top_cand["score"]
        texp = top_cand["explanation"]
        print(f"Top Candidate: {tid}, Score: {tscore}")
        for attribute, detail in texp.items():
            print(f"  - {attribute}: similarity={detail['similarity']}, weight_frac={detail['weight_fraction']}, contribution={detail['weighted_contribution']}")
    else:
        print("No candidates found or no data loaded.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
