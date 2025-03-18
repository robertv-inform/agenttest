# agenttest
To create conversational ai agent
    def compute_candidate_score(self, jd_attrs, cand_attrs, weights, model, faiss_index=None, faiss_cache=None):
        total_score = 0.0
        explanation = ""
        
        #################
        # EXPERIENCE (Binary using Embedding) – Revised to show one consolidated explanation
        #################
        # Filter out degree-specific keys (excluding total_experience)
        degree_keys = [k for k in jd_attrs.get("experience", {}) 
                       if k != "total_experience" and jd_attrs["experience"][k].get("gte", 0) > 0]
        candidate_exp = cand_attrs.get("total_experience", 0)
        best_overall_score = 0.0
        best_exp_explanation = ""
        for deg in degree_keys:
            req_years = jd_attrs["experience"][deg]["gte"]
            # Get candidate degrees from education records
            candidate_degrees = [edu.get("degree", "") for edu in cand_attrs.get("education", [])]
            best_sim = 0.0
            best_candidate = ""
            for cand_deg in candidate_degrees:
                sim_val, _ = self.get_similarity(deg, cand_deg, model, faiss_index, faiss_cache)
                if sim_val > best_sim:
                    best_sim = sim_val
                    best_candidate = cand_deg
            # If candidate's experience is sufficient and similarity meets threshold, score is 1.0
            current_score = 1.0 if candidate_exp >= req_years and best_sim >= 0.7 else best_sim
            # Keep track of the best overall score and its explanation detail
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_exp_explanation = (f"For {deg} (min {req_years} yrs), candidate has {candidate_exp} yrs; "
                                        f"best match '{best_candidate}' with similarity {best_sim:.2f}.")
        exp_weighted = best_overall_score * weights.get("experience", 0)
        explanation += f"Experience summary: {best_exp_explanation} Final Experience Score: {best_overall_score} (Weighted: {exp_weighted:.2f}); "
        total_score += exp_weighted
        
        # ... (rest of compute_candidate_score code remains unchanged)
