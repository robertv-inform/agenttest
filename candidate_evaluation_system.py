import os
import json
import pandas as pd
import numpy as np
import pickle
import traceback
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import openai
import re

########################################
# 1. ConfigManager
########################################
class ConfigManager:
    def __init__(self):
        self.ROOT_DIR = os.getcwd()
        self.CONFIG_DIR = os.path.join(self.ROOT_DIR, "config")
        self.DATA_DIR = os.path.join(self.ROOT_DIR, "data")
        self.JOB_DESC_DIR = os.path.join(self.ROOT_DIR, "jobdescription")
        self.LOG_DIR = os.path.join(self.ROOT_DIR, "log")
        self.CACHE_DIR = os.path.join(self.ROOT_DIR, "cache")
        self.MODEL_CACHE_DIR = os.path.join(self.ROOT_DIR, "model_cache")
        for folder in [self.CONFIG_DIR, self.DATA_DIR, self.JOB_DESC_DIR, self.LOG_DIR, self.CACHE_DIR, self.MODEL_CACHE_DIR]:
            os.makedirs(folder, exist_ok=True)
        self.JOB_DESC_FILE = "ms_Director__AppliedScience_jd1.txt"
        self.JOB_DESC_BASENAME = os.path.splitext(self.JOB_DESC_FILE)[0]
        self.KEY_ATTR_FILE = os.path.join(self.CONFIG_DIR, f"{self.JOB_DESC_BASENAME}_kab.json")
        self.WEIGHT_CONFIG_FILE = os.path.join(self.CONFIG_DIR, "keyattribute_weights.json")
        self.RESUME_CSV_FILE = os.path.join(self.DATA_DIR, "final_candidate_resumes_complete.csv")
        self.OUTPUT_LOG = os.path.join(self.LOG_DIR, "ranked_candidates_with_explanation.csv")

########################################
# 2. KeyAttributeExtractor
########################################
class KeyAttributeExtractor:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager

    def call_gpt4o_for_attributes(self, job_desc_text):
        try:
            tk_raw_skills = tk_jobparser_api()
        except Exception as e:
            print(f"Error in JD parsing using TK: {e}. \n")
            tk_raw_skills = []
          
        try:
            
            prompt = f"""
            SYSTEM INSTRUCTIONS:

            Translation to English:
            If the job description is in multiple languages or not in English, translate the entire text into English first.

            Strict Extraction Rules:
            Extract and normalize all relevant job attributes without making assumptions.
            Capture experience requirements correctly. For experience, return a field "experience" that includes:
              - "total_experience": the overall minimum years required, formatted as {{"gte": X}},
              - For each degree level (e.g., "Bachelor's", "Master's", "Doctorate"), return the minimum years required, formatted as {{"gte": X}}.
            For education, return an array where each object contains:
              - "preferred _degree_if_any": if a prepared degree is explicitly asked, otherwise an empty string,
              - "Normalized_degree": the normalized form of the prepared degree (e.g., "Doctorate"),
              - "degree": the required degree level (e.g., "Bachelor's", "Master's", "Doctorate"),
              - "abbreviation": the common short form (e.g., "BSc", "MSc", "PhD"),
              - "fields": a list of relevant fields; if not specified, use ["Computer Science", "Econometrics"].
            For skills, certifications, and languages, return lists of the required items.
            For location, return a JSON object with keys "city" and "country".  
               - "city": List the specific cities mentioned in the job description, comma-separated.
               - "country": The country name; if not specified, use an empty string.
            Strict JSON Formatting Guidelines:
            - Return only the JSON response.
            - The JSON must be strictly formatted with an indentation of 4 spaces.
            - If a field is not available, return it as an empty string ("") or an empty list ([]).

            Process the Following Job Description:
            {job_desc_text}
            
            - Note: The following skills have been pre-processed by our TK parser and should be used exactly as provided:  {tk_raw_skills}. Do not modify or re-order these skills. If you find any additional skills in the job description that are not already in the list, add them to the final skills list.
            

            JSON Output Format:
            {{
                "location": {{
                     "city": "New York, Boston",
                     "country": "United States"
                }},
                "experience": {{
                     "total_experience": {{"gte": 8}},
                     "Bachelor's": {{"gte": 8}},
                     "Master's": {{"gte": 6}},
                     "Doctorate": {{"gte": 5}}
                }},
                "skills": ["AI", "Machine Learning", "Natural Language Processing", "Management", "Data Science"],
                "education": [
                    {{
                        "preferred _degree_if_any": "Phd",
                        "Normalized_degree": "Doctorate"
                    }},
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
                "certifications": [],
                "languages": ["English"]
            }}

            Final Instructions for GPT-4o:
            Extract all job attributes exactly as per the format above.
            """
            # For demonstration, return a mock response:
            mock_response_json = {
                "location": ["New York, Boston"],
                "experience": {
                    "total_experience": {"gte": 8},
                    "Bachelor's": {"gte": 8},
                    "Master's": {"gte": 6},
                    "Doctorate": {"gte": 5}
                },
                "skills": ["AI", "Machine Learning", "Natural Language Processing", "Management", "Data Science"],
                "education": [
                    {
                        "preferred _degree_if_any": "Phd",
                        "Normalized_degree": "Doctorate"
                    },
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

    def get_or_create_key_attributes(self):
        jd_path = os.path.join(self.config.JOB_DESC_DIR, self.config.JOB_DESC_FILE)
        if not os.path.exists(jd_path):
            print(f"JD file '{self.config.JOB_DESC_FILE}' not found. Using empty attributes.")
            return {
                "location": {"city": "", "country": ""},
                "experience": {"total_experience": {"gte": 0},
                               "Bachelor's": {"gte": 0},
                               "Master's": {"gte": 0},
                               "Doctorate": {"gte": 0}},
                "skills": [],
                "education": [],
                "certifications": [],
                "languages": []
            }
        with open(jd_path, "r", encoding="utf-8") as f:
            jd_text = f.read()
        extracted = self.call_gpt4o_for_attributes(jd_text)
        if extracted is not None:
            try:
                with open(self.config.KEY_ATTR_FILE, "w", encoding="utf-8") as fw:
                    json.dump(extracted, fw, indent=4)
                print(f"JD attributes extracted and saved to {self.config.KEY_ATTR_FILE}.")
                return extracted
            except Exception as e:
                print(f"Error writing to KEY_ATTR_FILE: {e}")
                return {}
        else:
            print("Extraction failed; using empty attributes.")
            return {}

########################################
# 3. ResumeParser
########################################
class ResumeParser:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager

    def parse_resume_data(self):
        if not os.path.exists(self.config.RESUME_CSV_FILE):
            raise FileNotFoundError(f"Resume CSV not found at: {self.config.RESUME_CSV_FILE}")
        df = pd.read_csv(self.config.RESUME_CSV_FILE)
        candidates = []
        for _, row in df.iterrows():
            candidate_id = row.get("candidate_id", f"cand_{_}")
            resume_str = row.get("Resume", "{}")
            try:
                resume_json = json.loads(resume_str)
            except Exception:
                resume_json = {}
            candidate = self._parse_resume_dict(resume_json, candidate_id)
            candidates.append(candidate)
        return candidates

    def parse_resume_json(self, resume_json: dict) -> dict:
        if isinstance(resume_json, str):
            try:
                resume_json = json.loads(resume_json)
            except Exception as e:
                raise ValueError("Invalid resume JSON") from e
        resume_json.pop("customArea", None)
        candidate_id = resume_json.get("candidate_id", "unknown")
        return self._parse_resume_dict(resume_json, candidate_id)

    def _parse_resume_dict(self, resume_json: dict, candidate_id: str) -> dict:
        personal = resume_json.get("personal", {})
        address = personal.get("address", {})
        candidate_city = address.get("city", "").strip()
        candidate_country = address.get("country", {}).get("description", "").strip()
        candidate_location = {"city": candidate_city, "country": candidate_country}
        total_exp = float(resume_json.get("summary", {}).get("totalExperienceYears", 0))
        # Extract skills
        skills_data = resume_json.get("skills", {})
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
        # Extract education – use localDescription for candidate degree; use degreeName for fields.
        education_list = []
        for deg in resume_json.get("educationHistory", {}).get("degrees", []):
            cand_degree = deg.get("localDescription", "").strip()
            if not cand_degree:
                cand_degree = deg.get("degreeName", "").strip()
            cand_fields = deg.get("degreeName", "").strip().lower() if deg.get("degreeName") else ""
            if cand_degree:
                education_list.append({"degree": cand_degree.lower(), "fields": cand_fields})
        # Extract certifications
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
        # Extract languages
        languages_list = []
        if resume_json.get("lang"):
            languages_list.append(resume_json.get("lang").lower())
        return {
            "candidate_id": candidate_id,
            "location": candidate_location,
            "skills": skills_list,
            "education": education_list,
            "certifications": certifications_list,
            "languages": languages_list,
            "total_experience": total_exp
        }

########################################
# 4. EmbeddingManager
########################################
class EmbeddingManager:
    def __init__(self, config_manager: ConfigManager, use_faiss=False):
        self.config = config_manager
        self.USE_FAISS = use_faiss
        self.embedding_cache = {}
        if self.USE_FAISS:
            import faiss
            self.faiss = faiss
            self.DIMENSION = 384
            self.FAISS_INDEX_FILE = os.path.join(self.config.CACHE_DIR, "faiss_embedding_cache.pkl")
        else:
            self.faiss = None

    def get_model(self):
        model_path = os.path.join(self.config.MODEL_CACHE_DIR, "all-MiniLM-L6-v2")
        if not os.path.exists(model_path):
            print("Model not found locally. Downloading...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            model.save(model_path)
        else:
            model = SentenceTransformer(model_path)
        return model

    def simple_embed_text(self, text, model):
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        emb = model.encode(text).astype(np.float32)
        self.embedding_cache[text] = emb
        return emb

    def get_embedding(self, text, model, faiss_index=None, faiss_cache=None):
        if self.USE_FAISS:
            if text in faiss_cache:
                idx = faiss_cache.index(text)
                return faiss_index.reconstruct(idx)
            emb = model.encode(text).astype(np.float32)
            faiss_index.add(np.array([emb]))
            faiss_cache.append(text)
            with open(self.FAISS_INDEX_FILE, "wb") as f:
                pickle.dump({"index": faiss_index, "texts": faiss_cache}, f)
            return emb
        else:
            return self.simple_embed_text(text, model)

########################################
# 5. CandidateScorer
########################################
class CandidateScorer:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager

    # Use embedding similarity for all comparisons.
    def get_similarity(self, text1, text2, model, faiss_index=None, faiss_cache=None, threshold=0.85):
        emb1 = self.embedding_manager.get_embedding(text1.lower(), model, faiss_index, faiss_cache)
        emb2 = self.embedding_manager.get_embedding(text2.lower(), model, faiss_index, faiss_cache)
        sim = util.pytorch_cos_sim(emb1, emb2).item()
        if text1.lower() in text2.lower() or text2.lower() in text1.lower():
            return 1.0, f"Substring match between '{text1}' and '{text2}'"
        if sim >= threshold:
            return 1.0, f"Matched '{text1}' with '{text2}' yielding {sim:.2f} (threshold applied)"
        return sim, f"Matched '{text1}' with '{text2}' yielding {sim:.2f}"

    def multi_value_similarity(self, jd_list, cand_list, model, attribute_label="Attribute", faiss_index=None, faiss_cache=None):
        if not jd_list or not cand_list:
            return 0.0, ""
        best_scores = []
        details = ""
        for jd_item in jd_list:
            best_score_for_jd = 0.0
            best_candidate_for_jd = ""
            for cand_item in cand_list:
                sim, sim_detail = self.get_similarity(jd_item, cand_item, model, faiss_index, faiss_cache)
                if sim > best_score_for_jd:
                    best_score_for_jd = sim
                    best_candidate_for_jd = cand_item
            best_scores.append(best_score_for_jd)
            details += f"{attribute_label} match: For JD '{jd_item}', best candidate match '{best_candidate_for_jd}' yields {best_score_for_jd:.2f}; "
        overall_sim = np.mean(best_scores) if best_scores else 0.0
        explanation = f"{attribute_label} overall average similarity: {overall_sim:.2f}. Details: {details}"
        return overall_sim, explanation

    def single_value_similarity(self, jd_val, cand_val, model, faiss_index=None, faiss_cache=None):
        if not jd_val or not cand_val:
            return 0.0, ""
        if isinstance(jd_val, list):
            jd_val = " ".join(jd_val)
        if isinstance(cand_val, list):
            cand_val = " ".join(cand_val)
        sim, explanation = self.get_similarity(jd_val, cand_val, model, faiss_index, faiss_cache)
        return sim, explanation

    def load_weights(self):
        weight_file = self.embedding_manager.config.WEIGHT_CONFIG_FILE
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"No weight config found at {weight_file}")
        with open(weight_file, "r", encoding="utf-8") as f:
            weight_data = json.load(f)
        total = sum(weight_data.values())
        for key in weight_data:
            weight_data[key] = weight_data[key] / total
        return weight_data

    def compute_candidate_score(self, jd_attrs, cand_attrs, weights, model, faiss_index=None, faiss_cache=None):
        total_score = 0.0
        explanation = ""
        
        #################
        # EXPERIENCE (Binary using Embedding)
        #################
        degree_keys = [k for k in jd_attrs.get("experience", {}) 
                       if k != "total_experience" and jd_attrs["experience"][k].get("gte", 0) > 0]
        candidate_exp = cand_attrs.get("total_experience", 0)
        exp_met = False
        exp_explanation_lines = []
        if degree_keys:
            candidate_degrees = [edu.get("degree", "") for edu in cand_attrs.get("education", [])]
            for deg in degree_keys:
                req_years = jd_attrs["experience"][deg]["gte"]
                best_sim = 0.0
                best_candidate = ""
                for cand_deg in candidate_degrees:
                    sim_val, _ = self.get_similarity(deg, cand_deg, model, faiss_index, faiss_cache)
                    if sim_val > best_sim:
                        best_sim = sim_val
                        best_candidate = cand_deg
                exp_explanation_lines.append(
                    f"For {deg} (min {req_years} yrs): candidate has {candidate_exp} yrs; best match '{best_candidate}' similarity {best_sim:.2f}."
                )
                if candidate_exp >= req_years and best_sim >= 0.7:
                    exp_met = True
            overall_exp = 1.0 if exp_met else 0.0
            exp_weighted = overall_exp * weights.get("experience", 0)
            explanation += ("Experience details: " + " ".join(exp_explanation_lines) +
                            f" Final Experience Score: {overall_exp} (Weighted contribution: {exp_weighted:.2f}); ")
            total_score += exp_weighted
        else:
            required_total = jd_attrs.get("experience", {}).get("total_experience", {}).get("gte", 0)
            overall_exp = 1.0 if candidate_exp >= required_total else 0.0
            exp_weighted = overall_exp * weights.get("experience", 0)
            explanation += (f"Experience: Candidate has {candidate_exp} yrs, required {required_total}; " +
                            f" Final Experience Score: {overall_exp} (Weighted contribution: {exp_weighted:.2f}); ")
            total_score += exp_weighted
        
        #################
        # EDUCATION (Degree + Field Matching Revised)
        #################
        jd_edu = jd_attrs.get("education", [])
        candidate_edu = cand_attrs.get("education", [])
        if candidate_edu and isinstance(candidate_edu[0], str):
            candidate_edu = [{"degree": edu, "fields": ""} for edu in candidate_edu]
        
        edu_details = ""
        edu_score = 0.0
        prepared_entry = None
        other_entries = []
        for entry in jd_edu:
            if "preferred _degree_if_any" in entry and entry["preferred _degree_if_any"].strip():
                prepared_entry = entry
            else:
                other_entries.append(entry)
        
        # Helper for degree matching using embedding.
        def degree_match_score(jd_deg, cand_deg):
            if jd_deg in cand_deg or cand_deg in jd_deg:
                return 0.5, 1.0, "(substring match: full score 0.5)"
            sim, detail = self.get_similarity(jd_deg, cand_deg, model, faiss_index, faiss_cache)
            if sim >= 0.85:
                return 0.5, sim, f"(full degree match, sim {sim:.2f})"
            else:
                return sim * 0.5, sim, f"(partial degree match, sim {sim:.2f})"
        
        # Helper for field matching: compare each JD field individually against candidate's field.
        def field_match_score(jd_field, cand_field):
            sim, detail = self.get_similarity(jd_field, cand_field, model, faiss_index, faiss_cache)
            if sim >= 0.6:
                return 0.5, sim, f"(full field match, sim {sim:.2f})"
            else:
                return sim, sim, f"(partial field match, sim {sim:.2f})"
        
        # Helper to determine candidate degree level.
        def candidate_degree_level(cand_deg_str):
            s = cand_deg_str.lower()
            if "doctor" in s or "phd" in s:
                return "doctorate"
            elif "master" in s:
                return "master"
            elif "bachelor" in s:
                return "bachelor"
            return ""
        
        if prepared_entry:
            jd_prepared = prepared_entry.get("Normalized_degree", "").strip().lower()
            best_prepared_deg_score = 0.0
            best_prepared_sim = 0.0
            degree_explanation = ""
            for cand in candidate_edu:
                cand_local = cand.get("degree", "").strip().lower()
                score, sim_val, deg_detail = degree_match_score(jd_prepared, cand_local)
                degree_explanation += f"Comparing JD prepared '{jd_prepared}' vs candidate '{cand_local}' yields {deg_detail}; "
                if score > best_prepared_deg_score:
                    best_prepared_deg_score = score
                    best_prepared_sim = sim_val
            # Adjust degree score based on candidate degree level.
            cand_levels = [candidate_degree_level(cand.get("degree", "")) for cand in candidate_edu]
            if jd_prepared == "doctorate":
                if "doctorate" in cand_levels:
                    adjusted_degree_score = 0.5
                elif "master" in cand_levels:
                    adjusted_degree_score = 0.4
                elif "bachelor" in cand_levels:
                    adjusted_degree_score = 0.25
                else:
                    adjusted_degree_score = best_prepared_deg_score
            else:
                adjusted_degree_score = best_prepared_deg_score
            edu_details += f"Prepared degree match (JD '{jd_prepared}'): Degree Score {adjusted_degree_score:.2f} (raw sim {best_prepared_sim:.2f}); {degree_explanation}"
            # Field matching: check each JD field individually with candidate's fields.
            jd_fields = prepared_entry.get("fields", [])
            best_field_score = 0.0
            best_field_sim = 0.0
            field_explanation = ""
            if jd_fields:
                for jd_field in jd_fields:
                    for cand in candidate_edu:
                        cand_field = cand.get("fields", "").strip().lower()
                        score, sim_val, field_detail = field_match_score(jd_field.lower(), cand_field)
                        field_explanation += f"Comparing JD field '{jd_field}' vs candidate field '{cand_field}' yields {field_detail}; "
                        best_field_score = max(best_field_score, score)
                        best_field_sim = max(best_field_sim, sim_val)
            edu_details += f"Field match: Score {best_field_score:.2f} (raw sim {best_field_sim:.2f}); {field_explanation}"
            edu_score = adjusted_degree_score + best_field_score
        else:
            candidate_entry_scores = []
            edu_details_list = []
            for req in other_entries:
                jd_deg = req.get("degree", "").strip().lower()
                best_deg_score = 0.0
                best_deg_sim = 0.0
                deg_details = ""
                for cand in candidate_edu:
                    cand_local = cand.get("degree", "").strip().lower()
                    score, sim_val, deg_detail = degree_match_score(jd_deg, cand_local)
                    if score > best_deg_score:
                        best_deg_score = score
                        best_deg_sim = sim_val
                    deg_details += f"JD '{jd_deg}' vs candidate '{cand_local}': {deg_detail}; "
                jd_fields = req.get("fields", [])
                best_field_score = 0.0
                best_field_sim = 0.0
                field_details = ""
                if jd_fields:
                    for jd_field in jd_fields:
                        for cand in candidate_edu:
                            cand_field = cand.get("fields", "").strip().lower()
                            score, sim_val, field_detail = field_match_score(jd_field.lower(), cand_field)
                            if score > best_field_score:
                                best_field_score = score
                                best_field_sim = sim_val
                            field_details += f"JD field '{jd_field}' vs candidate '{cand_field}': {field_detail}; "
                candidate_entry_scores.append(best_deg_score + best_field_score)
                edu_details_list.append(f"JD '{jd_deg}': Degree score {best_deg_score:.2f} (sim {best_deg_sim:.2f}), Field score {best_field_score:.2f} (sim {best_field_sim:.2f}); Details: {deg_details} {field_details}")
            if candidate_entry_scores:
                edu_score = max(candidate_entry_scores)
                edu_details += " ".join(edu_details_list)
        overall_edu = min(edu_score, 1.0)
        edu_weighted = overall_edu * weights.get("education", 0)
        total_score += edu_weighted
        explanation += f"Education summary: {edu_details} yields weighted education similarity of {edu_weighted:.2f}; "
        
        #################
        # LOCATION
        #################
        jd_loc = jd_attrs.get("location", {})
        cand_loc = cand_attrs.get("location", {})
        if isinstance(jd_loc, dict) and isinstance(cand_loc, dict):
            jd_city = jd_loc.get("city", "").lower().strip()
            cand_city = cand_loc.get("city", "").lower().strip() if cand_loc.get("city") else ""
            jd_country = jd_loc.get("country", "").lower().strip()
            cand_country = cand_loc.get("country", "").lower().strip() if cand_loc.get("country") else ""
            sim_city, _ = self.get_similarity(jd_city, cand_city, model, faiss_index, faiss_cache)
            sim_country, _ = self.get_similarity(jd_country, cand_country, model, faiss_index, faiss_cache)
            loc_sim = (sim_city + sim_country) / 2.0 if (sim_city and sim_country) else max(sim_city, sim_country)
            explanation += f"Location: City similarity {sim_city:.2f}, Country similarity {sim_country:.2f} yields {loc_sim:.2f}; "
        else:
            loc_sim, loc_explanation = self.multi_value_similarity(
                jd_attrs.get("location", []),
                cand_attrs.get("location", []),
                model,
                attribute_label="Location",
                faiss_index=faiss_index,
                faiss_cache=faiss_cache
            )
            explanation += loc_explanation
        loc_weighted = loc_sim * weights.get("location", 0)
        total_score += loc_weighted
        explanation += f"Location weighted contribution: {loc_weighted:.2f}; "
        
        #################
        # SKILLS, CERTIFICATIONS, LANGUAGES
        #################
        for attr in ["skills", "certifications", "languages"]:
            jd_value = jd_attrs.get(attr, [])
            cand_value = cand_attrs.get(attr, [])
            if attr == "certifications" and not jd_value:
                explanation += f"Certifications: No requirement specified; weighted contribution: 0.00; "
                total_score += 0.0
                continue
            sim, attr_explanation = self.multi_value_similarity(
                jd_value, cand_value, model,
                attribute_label=attr.capitalize(),
                faiss_index=faiss_index,
                faiss_cache=faiss_cache
            )
            weighted_sim = sim * weights.get(attr, 0)
            total_score += weighted_sim
            explanation += f"{attr.capitalize()} weighted contribution: {weighted_sim:.2f}; {attr.capitalize()} details: {attr_explanation} "
        
        explanation += f"Overall Candidate Score (sum of weighted contributions): {total_score:.2f}."
        return total_score, explanation

########################################
# 6. CandidateEvaluationSystem (CSV-based flow)
########################################
class CandidateEvaluationSystem:
    def __init__(self):
        self.config_manager = ConfigManager()
        self.key_attribute_extractor = KeyAttributeExtractor(self.config_manager)
        self.resume_parser = ResumeParser(self.config_manager)
        self.embedding_manager = EmbeddingManager(self.config_manager, use_faiss=False)
        self.candidate_scorer = CandidateScorer(self.embedding_manager)
        openai.api_key = "YOUR_OPENAI_API_KEY"

    def run(self):
        try:
            print("Extracting JD key attributes...")
            jd_key_attrs = self.key_attribute_extractor.get_or_create_key_attributes()
            print("JD Attributes:")
            print(json.dumps(jd_key_attrs, indent=4))
            
            print("Parsing candidate resumes from CSV...")
            candidates = self.resume_parser.parse_resume_data()
            print(f"Found {len(candidates)} candidates.")
            
            print("Loading weights...")
            weights = self.candidate_scorer.load_weights()
            print("Loading model...")
            model = self.embedding_manager.get_model()
            
            faiss_index = None
            faiss_cache = None
            
            results = []
            for cand in candidates:
                score, exp = self.candidate_scorer.compute_candidate_score(
                    jd_key_attrs, cand, weights, model, faiss_index, faiss_cache
                )
                cand["score"] = score
                cand["explanation"] = exp
                results.append(cand)
            df = pd.DataFrame(results)
            df = df.sort_values(by="score", ascending=False)
            print("Ranked Candidates:")
            print(df[["candidate_id", "score", "explanation"]])
            
            df.to_csv(self.config_manager.OUTPUT_LOG, index=False)
            print(f"Results saved to {self.config_manager.OUTPUT_LOG}")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

########################################
# 7. ExecutiveExplanationFormatter
########################################
class ExecutiveExplanationFormatter:
    @staticmethod
    def format_explanation(raw_explanation: str) -> str:
        keys = ["Experience", "Education", "Location", "Skills", "Certifications", "Languages"]
        sections = {key: [] for key in keys}
        general = []
        current_key = None
        
        segments = [seg.strip() for seg in raw_explanation.split(";") if seg.strip()]
        for seg in segments:
            matched = False
            for key in keys:
                if re.match(rf'^{key}', seg, re.IGNORECASE):
                    current_key = key
                    sections[key].append(seg)
                    matched = True
                    break
            if not matched:
                if current_key:
                    sections[current_key].append(seg)
                else:
                    general.append(seg)
        html_parts = []
        if general:
            general_html = "<ul>" + "".join(f"<li>{ExecutiveExplanationFormatter.clean_segment(item)}</li>" for item in general) + "</ul>"
            html_parts.append(f"<li><strong>General:</strong> {general_html}</li>")
        for key in keys:
            if sections[key]:
                section_html = "<ul>" + "".join(f"<li>{ExecutiveExplanationFormatter.clean_segment(item)}</li>" for item in sections[key]) + "</ul>"
                html_parts.append(f"<li><strong>{key}:</strong> {section_html}</li>")
        final_html = (
            "<details style='margin-top:10px;'>"
            "<summary style='cursor:pointer; font-weight:bold;'>Executive Summary (click to expand)</summary>"
            "<ul>" + "".join(html_parts) + "</ul>"
            "</details>"
        )
        return final_html

    @staticmethod
    def clean_segment(segment: str) -> str:
        segment = segment.replace("-> sim:", "produces a similarity score of")
        segment = segment.replace("cosine", "")
        if segment:
            segment = segment[0].upper() + segment[1:]
        return segment

########################################
# 8. CandidateScoreService (New Functionality)
########################################
class CandidateScoreService:
    def __init__(self, use_faiss=False):
        self.config_manager = ConfigManager()
        self.resume_parser = ResumeParser(self.config_manager)
        self.embedding_manager = EmbeddingManager(self.config_manager, use_faiss=use_faiss)
        self.candidate_scorer = CandidateScorer(self.embedding_manager)
        self.weights = self.candidate_scorer.load_weights()
        self.model = self.embedding_manager.get_model()
        self.faiss_index = None
        self.faiss_cache = None
        openai.api_key = "YOUR_OPENAI_API_KEY"
        self.explanation_formatter = ExecutiveExplanationFormatter()

    def score_candidate(self, kab_config, resume_json):
        if isinstance(kab_config, str):
            try:
                kab_config = json.loads(kab_config)
            except Exception as e:
                raise ValueError("Invalid kab_config JSON") from e
        if isinstance(resume_json, str):
            try:
                resume_json = json.loads(resume_json)
            except Exception as e:
                raise ValueError("Invalid resume JSON") from e
        candidate = self.resume_parser.parse_resume_json(resume_json)
        score, raw_explanation = self.candidate_scorer.compute_candidate_score(
            kab_config, candidate, self.weights, self.model, self.faiss_index, self.faiss_cache
        )
        improved_explanation = self.explanation_formatter.format_explanation(raw_explanation)
        return score, improved_explanation

########################################
# Main Section
########################################
if __name__ == "__main__":
    # Option 1: CSV-Based Processing
    # system = CandidateEvaluationSystem()
    # system.run()

    # Option 2: Single Candidate JSON Processing
    service = CandidateScoreService()
    kab_config = KeyAttributeExtractor(service.config_manager).get_or_create_key_attributes()
    with open("64c43c61.json", "r", encoding="utf-8") as f:
        sample_resume = json.load(f)
    score, explanation = service.score_candidate(kab_config, sample_resume)
    print("\nNew Flow - Single Candidate JSON Scoring:")
    print(f"Candidate Score: {score}")
    print("Executive Explanation:")
    print(explanation)
