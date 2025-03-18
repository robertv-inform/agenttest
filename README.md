# agenttest
To create conversational ai agent
def multi_value_similarity(self, jd_list, cand_list, model, attribute_label="Attribute", faiss_index=None, faiss_cache=None):
    """
    Computes the similarity for an attribute. For the "skills" attribute,
    it applies a threshold-based scoring: for each required skill, if the best matching
    candidate skill has a similarity >= threshold (e.g., 0.7), it counts as a full match (score 1),
    otherwise 0. The overall score is the fraction of required skills fully matched.
    
    For other attributes, it uses the arithmetic mean of the best similarity scores.
    """
    if not jd_list or not cand_list:
        return 0.0, ""
    
    if attribute_label.lower() == "skills":
        THRESHOLD = 0.7
        match_count = 0
        details = ""
        for jd_item in jd_list:
            best_sim = 0.0
            for cand_item in cand_list:
                sim, _ = self.get_similarity(jd_item, cand_item, model, faiss_index, faiss_cache)
                best_sim = max(best_sim, sim)
            if best_sim >= THRESHOLD:
                match_count += 1
                details += f"Skill '{jd_item}': match (sim {best_sim:.2f}); "
            else:
                details += f"Skill '{jd_item}': no match (sim {best_sim:.2f}); "
        overall_sim = match_count / len(jd_list)
        explanation = f"Skills matched: {match_count}/{len(jd_list)}. {details}"
        return overall_sim, explanation
    else:
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
            details += f"{attribute_label} - JD '{jd_item}' best match '{best_candidate_for_jd}' sim {best_score_for_jd:.2f}; "
        overall_sim = np.mean(best_scores)
        explanation = f"{attribute_label} average similarity: {overall_sim:.2f}. {details}"
        return overall_sim, explanation


@staticmethod
def format_explanation(raw_explanation: str) -> str:
    """
    Produces a simplified executive summary from the raw explanation.
    It extracts one summary line per attribute based on known keywords and
    returns a concise bullet-point summary.
    """
    # Define keys of interest.
    keys = ["Experience", "Education", "Location", "Skills", "Certifications", "Languages"]
    summary_lines = {}
    # Split explanation by semicolons.
    segments = [seg.strip() for seg in raw_explanation.split(";") if seg.strip()]
    # For each segment, if it starts with one of the keys, take it as the summary line.
    for seg in segments:
        for key in keys:
            if seg.lower().startswith(key.lower()):
                # Extract the first sentence (up to the first period) for brevity.
                sentence = seg.split(".")[0]
                summary_lines[key] = sentence
                break
    # Build the final summary.
    final_summary = "Executive Summary:\n"
    for key in keys:
        if key in summary_lines:
            final_summary += f"- {key}: {summary_lines[key].strip()}.\n"
    return final_summary
