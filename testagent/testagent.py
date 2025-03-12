def compute_candidate_score(jd_attrs, cand_attrs, weights, model, faiss_index=None, faiss_cache=None):
    """
    Compute the candidate score by comparing each attribute.
    
    EXPERIENCE:
    - If the JD's "experience" contains degree-specific requirements (keys besides "total_experience"),
      then check if the candidate's total experience meets at least one of these degree requirements.
      If so, experience score = 1; otherwise, 0.
    - If no degree-specific keys are provided, only use the overall "total_experience" threshold.
    
    EDUCATION:
    - For each required education entry in the JD, construct a string from the required degree and fields.
    - For each candidate education string (from localDescription or degreeName), compute the cosine similarity
      using SentenceTransformer embeddings.
    - If any candidate education has a similarity >= 0.7 for any JD education requirement, set the education score to 1.
      Otherwise, set it to 0.
    
    OTHER ATTRIBUTES:
    - For location, skills, certifications, and languages, use embedding-based similarity (or structured matching for location).
    
    The final candidate score is the weighted sum of each attribute's match score.
    """
    total_score = 0.0
    explanation = ""
    
    # EXPERIENCE:
    jd_exp = jd_attrs.get("experience", {})
    candidate_exp = cand_attrs.get("total_experience", 0)
    # Check if degree-specific keys exist (keys other than "total_experience")
    degree_keys = [k for k in jd_exp.keys() if k != "total_experience"]
    if degree_keys:
        exp_match = False
        for deg in degree_keys:
            req_years = jd_exp.get(deg, {}).get("gte", 0)
            # Check if candidate's education (normalized) contains the degree name
            candidate_edu = [edu.lower() for edu in cand_attrs.get("education", [])]
            if any(deg.lower() in edu for edu in candidate_edu) and (candidate_exp >= req_years):
                exp_match = True
                explanation += f"Experience: candidate's exp {candidate_exp} meets {deg} requirement (gte {req_years}); "
                break
            else:
                explanation += f"Experience: {deg} requirement (gte {req_years}) not met; "
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
    # Get candidate education strings (normalized)
    candidate_edu = [edu.lower() for edu in cand_attrs.get("education", [])]
    # For each required education, construct a string: "degree, field1, field2, ..."
    for req in edu_reqs:
        req_degree = req.get("degree", "").lower()
        req_fields = [f.lower() for f in req.get("fields", [])]
        jd_edu_str = req_degree
        if req_fields:
            jd_edu_str += ", " + ", ".join(req_fields)
        # For each candidate education, compute cosine similarity
        max_sim = 0.0
        best_match = ""
        for edu in candidate_edu:
            # Compute embeddings for JD requirement and candidate education string
            jd_emb = get_embedding(jd_edu_str, model, faiss_index, faiss_cache)
            cand_emb = get_embedding(edu, model, faiss_index, faiss_cache)
            sim_val = util.pytorch_cos_sim(jd_emb, cand_emb).item()
            if sim_val > max_sim:
                max_sim = sim_val
                best_match = edu
        edu_explanation += f"JD edu '{jd_edu_str}' vs candidate '{best_match}' -> sim: {max_sim:.2f}; "
        # Use OR logic: if any candidate education has similarity >= 0.7, education score = 1
        if max_sim >= 0.7:
            education_sim = 1.0
            break
    total_score += education_sim * weights.get("education", 0)
    explanation += f"Education -> {edu_explanation} weighted sim: {education_sim * weights.get('education', 0):.2f}; "
    
    # LOCATION: Compare structured location
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
    
    # OTHER ATTRIBUTES: skills, certifications, languages using embedding-based similarity.
    for attr in ["skills", "certifications", "languages"]:
        jd_value = jd_attrs.get(attr, [])
        cand_value = cand_attrs.get(attr, [])
        sim, attr_explanation = multi_value_similarity(jd_value, cand_value, model, faiss_index, faiss_cache)
        weighted_sim = sim * weights.get(attr, 0)
        total_score += weighted_sim
        explanation += f"{attr.capitalize()} -> {attr_explanation} weighted sim: {weighted_sim:.2f}; "
    
    return total_score, explanation
