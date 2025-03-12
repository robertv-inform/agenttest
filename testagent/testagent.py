def compute_candidate_score(jd_attrs, cand_attrs, weights, model, faiss_index=None, faiss_cache=None):
    """
    Compute the candidate score by comparing each attribute.
    
    EXPERIENCE:
      - If the JD's "experience" contains degree-specific keys (besides "total_experience"),
        for each required degree (e.g., "Bachelor's", "Master's", "Doctorate"), we compute the
        embedding similarity between the JD degree (using its text from the config) and each candidate
        education entry (from localDescription if available, else degreeName). A fallback substring check forces
        similarity = 1 if one string is contained in the other.
        If any candidate education yields a similarity >= 0.7 and the candidate's total experience meets the required years,
        then experience score = 1; otherwise, 0.
      - If no degree-specific keys exist, compare candidate's total experience against the overall threshold.
    
    EDUCATION:
      - For each JD education requirement, we separately compare:
           * The required degree (jd_degree) with the candidate’s degree part (extracted from localDescription).
           * Each required field (from jd_fields) with the candidate’s field information (extracted from degreeName).
      - We assume that if both localDescription and degreeName are available in a candidate education entry, they are separated by a comma.
      - For each candidate education entry:
           Let candidate_degree = part before comma (if present) or the whole string.
           Let candidate_fields = part after comma (if present) else empty.
      - Compute embedding cosine similarity for the degree and for each required field separately.
      - If the degree similarity >= 0.7 and at least one required field similarity >= 0.7, that's a full match (score = 1).
      - If only one of these meets the threshold, that yields a half match (score = 0.5).
      - Using OR logic across candidate education entries, if any yields a full match, education score = 1; if none full but at least one half, then 0.5; else 0.
    
    LOCATION, SKILLS, CERTIFICATIONS, LANGUAGES:
      - These are compared using embedding-based cosine similarity (or structured matching for location).
    
    The final candidate score is the weighted sum of each attribute's match score.
    """
    total_score = 0.0
    explanation = ""
    
    # EXPERIENCE:
    jd_exp = jd_attrs.get("experience", {})
    candidate_exp = cand_attrs.get("total_experience", 0)
    degree_keys = [k for k in jd_exp.keys() if k != "total_experience"]
    if degree_keys:
        exp_match = False
        for deg in degree_keys:
            req_years = jd_exp.get(deg, {}).get("gte", 0)
            candidate_edu = [edu.lower() for edu in cand_attrs.get("education", [])]
            jd_deg_text = deg.lower()
            max_sim = 0.0
            for edu in candidate_edu:
                jd_emb = get_embedding(jd_deg_text, model, faiss_index, faiss_cache)
                cand_emb = get_embedding(edu, model, faiss_index, faiss_cache)
                sim_val = util.pytorch_cos_sim(jd_emb, cand_emb).item()
                if jd_deg_text in edu or edu in jd_deg_text:
                    sim_val = max(sim_val, 1.0)
                if sim_val > max_sim:
                    max_sim = sim_val
            explanation += f"Experience: For {deg} (req gte {req_years}) degree sim = {max_sim:.2f}; "
            if max_sim >= 0.7 and candidate_exp >= req_years:
                exp_match = True
                explanation += f"Matched {deg} requirement (candidate exp {candidate_exp}); "
                break
        exp_sim = 1.0 if exp_match else 0.0
        total_score += exp_sim * weights.get("experience", 0)
    else:
        required_total = jd_exp.get("total_experience", {}).get("gte", 0)
        exp_sim = 1.0 if candidate_exp >= required_total else 0.0
        explanation += f"Total Experience: required gte {required_total}, candidate {candidate_exp} -> sim: {exp_sim}; "
        total_score += exp_sim * weights.get("experience", 0)
    
    # EDUCATION:
    edu_reqs = jd_attrs.get("education", [])
    education_sim = 0.0
    edu_explanation = ""
    candidate_edu = [edu.lower() for edu in cand_attrs.get("education", [])]
    req_scores = []
    for req in edu_reqs:
        jd_degree = req.get("degree", "").lower()
        jd_fields = req.get("fields", [])
        # We'll check each candidate education entry separately.
        for edu in candidate_edu:
            # Attempt to split the candidate education string by comma.
            parts = [p.strip() for p in edu.split(",")]
            if len(parts) >= 2:
                cand_deg = parts[0]
                cand_field_str = " ".join(parts[1:])  # concatenate the remaining parts
            else:
                cand_deg = parts[0]
                cand_field_str = ""
            # Compute embedding similarity for the degree portion:
            jd_emb_degree = get_embedding(jd_degree, model, faiss_index, faiss_cache)
            cand_emb_degree = get_embedding(cand_deg, model, faiss_index, faiss_cache)
            deg_sim = util.pytorch_cos_sim(jd_emb_degree, cand_emb_degree).item()
            if jd_degree in cand_deg or cand_deg in jd_degree:
                deg_sim = max(deg_sim, 1.0)
            # For fields, check each required field individually against candidate's degreeName portion (cand_field_str)
            field_matches = []
            for req_field in jd_fields:
                req_field = req_field.lower().strip()
                jd_emb_field = get_embedding(req_field, model, faiss_index, faiss_cache)
                cand_emb_field = get_embedding(cand_field_str, model, faiss_index, faiss_cache)
                field_sim = util.pytorch_cos_sim(jd_emb_field, cand_emb_field).item()
                if req_field in cand_field_str or cand_field_str in req_field:
                    field_sim = max(field_sim, 1.0)
                field_matches.append(field_sim)
            field_sim_max = max(field_matches) if field_matches else 0.0
            edu_explanation += f"JD edu '{jd_degree}' vs cand deg '{cand_deg}' -> sim: {deg_sim:.2f}; "
            edu_explanation += f"JD fields {jd_fields} vs cand fields '{cand_field_str}' -> max sim: {field_sim_max:.2f}; "
            # Determine match score:
            if deg_sim >= 0.7 and field_sim_max >= 0.7:
                req_scores.append(1.0)
            elif deg_sim >= 0.7 or field_sim_max >= 0.7:
                req_scores.append(0.5)
            else:
                req_scores.append(max(deg_sim, field_sim_max))
        # OR logic: if any candidate entry gets full match for this requirement, break.
        if any(score == 1.0 for score in req_scores):
            break
    if req_scores:
        if 1.0 in req_scores:
            education_sim = 1.0
        elif 0.5 in req_scores:
            education_sim = 0.5
        else:
            education_sim = max(req_scores)
    else:
        education_sim = 0.0
    total_score += education_sim * weights.get("education", 0)
    explanation += f"Education -> {edu_explanation} weighted sim: {education_sim * weights.get('education', 0):.2f}; "
    
    # LOCATION:
    jd_loc = jd_attrs.get("location", {})
    cand_loc = cand_attrs.get("location", {})
    if isinstance(jd_loc, dict) and isinstance(cand_loc, dict):
        city_match = False
        country_match = False
        if jd_loc.get("city"):
            jd_city = jd_loc.get("city").lower().strip()
            cand_city = cand_loc.get("city").lower().strip() if cand_loc.get("city") else ""
            city_match = (jd_city in cand_city) or (cand_city in jd_city) or (jd_city == cand_city)
        if jd_loc.get("country"):
            jd_country = jd_loc.get("country").lower().strip()
            cand_country = cand_loc.get("country").lower().strip() if cand_loc.get("country") else ""
            country_match = (jd_country in cand_country) or (cand_country in jd_country) or (jd_country == cand_country)
        if jd_loc.get("city"):
            loc_sim = 1.0 if city_match else 0.0
            loc_explanation = f"City match: required '{jd_loc.get('city')}', candidate '{cand_loc.get('city')}'; "
        elif jd_loc.get("country"):
            loc_sim = 1.0 if country_match else 0.0
            loc_explanation = f"Country match: required '{jd_loc.get('country')}', candidate '{cand_loc.get('country')}'; "
        else:
            loc_sim = 0.0
            loc_explanation = "No location requirement provided; "
    else:
        loc_sim, loc_explanation = multi_value_similarity(jd_attrs.get("location", []), cand_attrs.get("location", []), model, faiss_index, faiss_cache)
    total_score += loc_sim * weights.get("location", 0)
    explanation += f"Location -> {loc_explanation} weighted sim: {loc_sim * weights.get('location', 0):.2f}; "
    
    # OTHER ATTRIBUTES: skills, certifications, languages.
    for attr in ["skills", "certifications", "languages"]:
        jd_value = jd_attrs.get(attr, [])
        cand_value = cand_attrs.get(attr, [])
        sim, attr_explanation = multi_value_similarity(jd_value, cand_value, model, faiss_index, faiss_cache)
        weighted_sim = sim * weights.get(attr, 0)
        total_score += weighted_sim
        explanation += f"{attr.capitalize()} -> {attr_explanation} weighted sim: {weighted_sim:.2f}; "
    
    return total_score, explanation
