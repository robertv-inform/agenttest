# agenttest
To create conversational ai agent
import re
import json

class ExecutiveExplanationFormatter:
    @staticmethod
    def format_explanation(raw_explanation: str) -> str:
        """
        Produces a plain, executive-friendly summary.
        It extracts a short summary for each key attribute (excluding Location)
        and returns a bullet-point summary in plain language.
        """
        # Define keys to include (exclude Location)
        keys = ["Experience", "Education", "Skills", "Certifications", "Languages"]
        summary = {}

        # For Experience: we assume raw_explanation contains a phrase like "Final Experience Score: X"
        if "Final Experience Score:" in raw_explanation:
            if "Final Experience Score: 1.0" in raw_explanation:
                summary["Experience"] = "Candidate fully meets the experience requirements."
            else:
                summary["Experience"] = "Candidate does not fully meet the experience requirements."
        else:
            summary["Experience"] = "Experience evaluation not available."

        # For Education: check if weighted education similarity is 1.00
        if "Education summary:" in raw_explanation:
            # Look for a pattern like "yields weighted education similarity of X"
            m = re.search(r"yields weighted education similarity of\s*([\d\.]+)", raw_explanation)
            if m and float(m.group(1)) >= 1.0:
                summary["Education"] = "Candidate's education fully meets the requirements."
            else:
                summary["Education"] = "Candidate's education is only partially matching requirements."
        else:
            summary["Education"] = "Education evaluation not available."

        # For Skills: We assume our threshold-based scoring prints "Skills overall match: X/Y fully matched."
        m = re.search(r"Skills overall match:\s*(\d+)/(\d+)", raw_explanation, re.IGNORECASE)
        if m:
            x = m.group(1)
            y = m.group(2)
            summary["Skills"] = f"Candidate fully matches {x} out of {y} required skills."
        else:
            summary["Skills"] = "Skills evaluation not available."

        # For Certifications: If no certifications are required, we output accordingly.
        if "Certifications:" in raw_explanation:
            if "No requirement" in raw_explanation:
                summary["Certifications"] = "No certification requirements."
            else:
                summary["Certifications"] = "Candidate meets certification requirements."
        else:
            summary["Certifications"] = "Certifications evaluation not available."

        # For Languages: A simple statement.
        if "Languages:" in raw_explanation:
            summary["Languages"] = "Candidate meets language requirements."
        else:
            summary["Languages"] = "Language evaluation not available."

        # Build the final plain summary as bullet points.
        final_lines = ["Executive Summary:"]
        for key in keys:
            final_lines.append(f"- {key}: {summary.get(key, 'N/A')}")
        return "\n".join(final_lines)

    @staticmethod
    def clean_segment(segment: str) -> str:
        # (Not used in this plain summary version.)
        segment = segment.replace("-> sim:", "produces a similarity score of")
        segment = segment.replace("cosine", "")
        if segment:
            segment = segment[0].upper() + segment[1:]
        return segment
