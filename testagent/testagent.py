def compute_candidate_score(jd_attrs, cand_attrs, weights, model, faiss_index=None, faiss_cache=None):
    """
    Compute the candidate score by comparing each attribute.

    EXPERIENCE:
      - If the JD's "experience" contains degree-specific keys (besides "total_experience"),
        for each required degree, compute the embedding similarity between the JD degree
        (e.g., "Bachelor's") and each candidate education entry (extracted from localDescription or degreeName).
        If any candidate education has similarity >= 0.7 and candidate's total experience is at or above the required threshold,
        then the experience score is 1; otherwise, 0.
      - If no degree-specific keys exist, simply compare candidate's total experience with the overall requirement.

    EDUCATION:
      - For each required education entry in the JD, construct a composite string (combining required degree and fields).
      - For each candidate education entry, compute the cosine similarity using SentenceTransformer embeddings.
      - If any candidate education yields a similarity >= 0.7, education score = 1; otherwise, 0.

    LOCATION, SKILLS, CERTIFICATIONS, LANGUAGES:
      - These are compared as before using embedding-based cosine similarity (or structured matching for location).

    The final score is the weighted sum of each attribute's match score.
    """
    total_score = 0.0
    explanation = ""
    
    # EXPERIENCE:
    jd_exp = jd_attrs.get("experience", {})
    candidate_exp = cand_attrs.get("total_experience", 0)
    # Check for degree-specific keys besides "total_experience"
    degree_keys = [k for k in jd_exp.keys() if k != "total_experience"]
    if degree_keys:
        exp_match = False
        for deg in degree_keys:
            req_years = jd_exp.get(deg, {}).get("gte", 0)
            # Use embedding to compare the JD required degree with candidate education
            candidate_edu = [edu.lower() for edu in cand_attrs.get("education", [])]
            jd_degree_str = deg.lower()
            max_sim = 0.0
            for edu in candidate_edu:
                jd_emb = get_embedding(jd_degree_str, model, faiss_index, faiss_cache)
                cand_emb = get_embedding(edu, model, faiss_index, faiss_cache)
                sim_val = util.pytorch_cos_sim(jd_emb, cand_emb).item()
                if sim_val > max_sim:
                    max_sim = sim_val
            explanation += f"Experience: For required {deg} (gte {req_years}), degree sim = {max_sim:.2f}; "
            if max_sim >= 0.7 and candidate_exp >= req_years:
                exp_match = True
                explanation += f"Matched {deg} requirement; "
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
    # Use candidate education strings already parsed (from localDescription or degreeName)
    candidate_edu = [edu.lower() for edu in cand_attrs.get("education", [])]
    for req in edu_reqs:
        req_degree = req.get("degree", "").lower()
        req_fields = [field.lower() for field in req.get("fields", [])]
        # Build a composite JD education string
        jd_edu_str = req_degree
        if req_fields:
            jd_edu_str += ", " + ", ".join(req_fields)
        max_sim = 0.0
        best_match = ""
        for edu in candidate_edu:
            jd_emb = get_embedding(jd_edu_str, model, faiss_index, faiss_cache)
            cand_emb = get_embedding(edu, model, faiss_index, faiss_cache)
            sim_val = util.pytorch_cos_sim(jd_emb, cand_emb).item()
            if sim_val > max_sim:
                max_sim = sim_val
                best_match = edu
        edu_explanation += f"JD edu '{jd_edu_str}' vs candidate '{best_match}' -> sim: {max_sim:.2f}; "
        if max_sim >= 0.7:
            education_sim = 1.0
            break
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
    
    # OTHER ATTRIBUTES: skills, certifications, languages via embedding-based similarity.
    for attr in ["skills", "certifications", "languages"]:
        jd_value = jd_attrs.get(attr, [])
        cand_value = cand_attrs.get(attr, [])
        sim, attr_explanation = multi_value_similarity(jd_value, cand_value, model, faiss_index, faiss_cache)
        weighted_sim = sim * weights.get(attr, 0)
        total_score += weighted_sim
        explanation += f"{attr.capitalize()} -> {attr_explanation} weighted sim: {weighted_sim:.2f}; "
    
    return total_score, explanation
