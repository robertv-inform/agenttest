import pandas

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
        explanation += ("Experience summary: " + " ".join(exp_explanation_lines) +
                        f" Final Experience Score: {overall_exp} (Weighted: {exp_weighted:.2f}); ")
        total_score += exp_weighted
    else:
        required_total = jd_attrs.get("experience", {}).get("total_experience", {}).get("gte", 0)
        overall_exp = 1.0 if candidate_exp >= required_total else 0.0
        exp_weighted = overall_exp * weights.get("experience", 0)
        explanation += (f"Experience: Candidate has {candidate_exp} yrs, required {required_total}; " +
                        f" Final Experience Score: {overall_exp} (Weighted: {exp_weighted:.2f}); ")
        total_score += exp_weighted

    #################
    # EDUCATION (Degree + Field Matching – Updated)
    #################
    jd_edu = jd_attrs.get("education", [])
    candidate_edu = cand_attrs.get("education", [])
    # candidate_edu is expected to be a list of dictionaries with keys:
    # "degree" (from localDescription) and "fields" (from degreeName).
    best_total = 0.0
    best_entry_details = ""
    # Process each candidate education entry
    for cand in candidate_edu:
        cand_local = cand.get("degree", "").strip().lower()  # Candidate degree from localDescription
        cand_full = cand.get("fields", "").strip().lower()    # Candidate full degree text (degreeName)
        entry_best = 0.0
        entry_details = ""
        # Compare this candidate entry with each JD education requirement
        for req in jd_edu:
            jd_deg = req.get("degree", "").strip().lower()  # JD required degree
            jd_fields = req.get("fields", [])
            # --- Degree Matching ---
            if jd_deg in cand_local or cand_local in jd_deg:
                degree_score = 0.5  # full half score
                degree_detail = f"Degree '{jd_deg}' matched fully with '{cand_local}' (score 0.5)."
            else:
                sim_deg, _ = self.get_similarity(jd_deg, cand_local, model, faiss_index, faiss_cache)
                degree_score = 0.5 if sim_deg >= 0.85 else sim_deg * 0.5
                degree_detail = f"Degree '{jd_deg}' similarity {sim_deg:.2f} gives score {degree_score:.2f}."
            # --- Field Matching ---
            # Remove candidate degree (cand_local) from the full candidate text (cand_full) to get the field information.
            if cand_local and cand_local in cand_full:
                field_text = cand_full.replace(cand_local, "").strip()
            else:
                field_text = cand_full
            field_score = 0.0
            field_detail = ""
            for jd_field in jd_fields:
                sim_field, _ = self.get_similarity(jd_field.lower(), field_text, model, faiss_index, faiss_cache)
                # Award full half score (0.5) if similarity >= 0.6; else proportional.
                temp_score = 0.5 if sim_field >= 0.6 else sim_field
                if temp_score > field_score:
                    field_score = temp_score
                    field_detail = f"Field '{jd_field}' similarity {sim_field:.2f} gives score {temp_score:.2f}."
            total_req_score = degree_score + field_score
            if total_req_score > entry_best:
                entry_best = total_req_score
                entry_details = f"{degree_detail} {field_detail}"
        if entry_best > best_total:
            best_total = entry_best
            best_entry_details = entry_details
    overall_edu = min(best_total, 1.0)
    edu_weighted = overall_edu * weights.get("education", 0)
    total_score += edu_weighted
    explanation += f"Education summary: {best_entry_details} yields weighted education similarity of {edu_weighted:.2f}; "
    
    #################
    # LOCATION
    #################
    jd_loc = jd_attrs.get("location", {})
    cand_loc = cand_attrs.get("location", {})
    if self.embedding_manager.config.ENABLE_LOCATION:
        if isinstance(jd_loc, dict) and isinstance(cand_loc, dict):
            jd_city = jd_loc.get("city", "").lower().strip()
            cand_city = cand_loc.get("city", "").lower().strip() if cand_loc.get("city") else ""
            jd_country = jd_loc.get("country", "").lower().strip()
            cand_country = cand_loc.get("country", "").lower().strip() if cand_loc.get("country") else ""
            sim_city, _ = self.get_similarity(jd_city, cand_city, model, faiss_index, faiss_cache)
            sim_country, _ = self.get_similarity(jd_country, cand_country, model, faiss_index, faiss_cache)
            loc_sim = (sim_city + sim_country) / 2.0 if (sim_city and sim_country) else max(sim_city, sim_country)
            explanation += f"Location: City sim {sim_city:.2f}, Country sim {sim_country:.2f} yields {loc_sim:.2f}; "
            loc_weighted = loc_sim * weights.get("location", 0)
            explanation += f"Location weighted contribution: {loc_weighted:.2f}; "
            total_score += loc_weighted
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
            explanation += f"Location weighted contribution: {loc_weighted:.2f}; "
            total_score += loc_weighted
    else:
        explanation += "Location disabled; "
    
    #################
    # SKILLS, CERTIFICATIONS, LANGUAGES
    #################
    for attr in ["skills", "certifications", "languages"]:
        jd_value = jd_attrs.get(attr, [])
        cand_value = cand_attrs.get(attr, [])
        if attr == "certifications" and not jd_value:
            explanation += f"Certifications: No requirement; weighted contribution: 0.00; "
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
    
    explanation += f"Overall Candidate Score: {total_score:.2f}."
    return total_score, explanation
    
import re

class ExecutiveExplanationFormatter:
    @staticmethod
    def format_explanation(raw_explanation: str) -> str:
        """
        Produces a plain, executive-friendly summary from the raw explanation.
        Only the following attributes are included: Experience, Education,
        Skills, Certifications, and Languages (Location is excluded).
        
        For each attribute, the formatter extracts the first sentence (up to the first period)
        from the corresponding raw explanation segment and builds a bullet-point summary.
        """
        # Define keys to include (excluding "Location")
        keys = ["Experience", "Education", "Skills", "Certifications", "Languages"]
        summary_lines = {}
        
        # For each key, search for the first segment starting with that key.
        for key in keys:
            pattern = rf"{key}:[^;]+"
            match = re.search(pattern, raw_explanation, re.IGNORECASE)
            if match:
                # Extract the first sentence (up to the first period)
                sentence = match.group(0).split(".")[0]
                summary_lines[key] = ExecutiveExplanationFormatter.clean_segment(sentence.strip())
        
        # Build the final summary as plain text bullet points.
        lines = ["Executive Summary:"]
        for key in keys:
            if key in summary_lines:
                lines.append(f"- {key}: {summary_lines[key]}.")
        return "\n".join(lines)

    @staticmethod
    def clean_segment(segment: str) -> str:
        """
        Cleans a segment by replacing technical phrases with plain text.
        """
        segment = segment.replace("-> sim:", "produces a similarity score of")
        segment = segment.replace("cosine", "")
        # Ensure the segment starts with an uppercase letter.
        if segment:
            segment = segment[0].upper() + segment[1:]
        return segment

