# agenttest
# Certificate Matching System - Pseudocode Documentation

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Certificate Matching System                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data Models   │  │   Retrievers    │  │    Matcher   │ │
│  │   - Certificate │  │   - Job Profile │  │   - Semantic │ │
│  │   - Employee    │  │   - Employee    │  │   - OpenAI   │ │
│  │   - JobProfile  │  │   - Factory     │  │   - Bidir.   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│                              │                              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Main Service Orchestrator                     │ │
│  │  - Coordinates retrieval and matching                   │ │
│  │  - Calculates scores and rankings                       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```
1. Data Models

Define class Certificate(name, level, category, expiry_date)
Define class Employee(id, name, certificates: list of Certificate)
Define class JobProfile(id, title, required_certificates: list of Certificate)
Define class BidirectionalMatch(job_cert, employee_cert, forward_similarity, reverse_similarity, avg_similarity, is_range_match, confidence_level)
Define class MatchResult(employee_id, employee_name, bidirectional_matches: list, total_score, match_pe


End-to-End Certification Matching – Step-by-Step Pseudocode
Step 1: Start with a Job Profile ID and an Employee ID.

Step 2: Retrieve the list of required certificates for the job profile using an API or data source.

Step 3: Retrieve the list of employee certificates for the employee using an API or data source.

Step 4: For each certificate in the job profile’s required list:

    - Add a prefix to the certificate text: "the professional certificate in: ".
    - Generate an embedding (vector) using OpenAI’s text-embedding-3-large model.

    For each certificate from the employee’s list:

        - Add the same prefix to the certificate text.
        - Generate the embedding.
        - Calculate cosine similarity between the job certificate embedding and the employee certificate embedding.
        - If the similarity is greater than or equal to 0.50, keep it as a valid match.

    - From the valid matches, select the top 2 employee certificates with the highest similarity scores.

Step 5: Do a reverse check for the top 2 matches:

    - Take each of the top 2 employee certificates.
    - Use the employee certificate as the source and compare it to the original job certificate (reverse direction).
    - Recalculate cosine similarity for each reverse match.
    - Rank them based on reverse scores.

Step 6: Choose the best match based on agreement between original and reverse directions:

    - If the top match in reverse is the same as in the forward direction, keep it.
    - If the second match in reverse is ranked higher, prefer that instead.

Step 7: Check the range between the top 2 scores:

    - If the difference between top 1 and top 2 similarity scores is between 0.10 and 0.50, accept both matches.
    - If the difference is more than 0.50, only consider the top match as valid.

Step 8: Record the results:

    - For each job certificate, store the best-matched employee certificate(s), similarity score(s), and matching status.

Step 9: Repeat Steps 4–8 for each required certificate in the job profile.

Step 10: Combine all matches into a result for that employee.

Step 11: If multiple employees are evaluated for one job profile, repeat Steps 2–10 for each employee.

Step 12: Return or store the final result per employee, which includes:

    - Employee ID
    - Job Profile ID
    - Matched certificate pairs (job cert ↔ employee cert)
    - Similarity scores
    - Notes on whether one or two matches were accepted for each

Note: This method uses pure semantic similarity (no abbreviation rules) and ensures accuracy through bidirectional embedding checks, score thresholding, and difference-based range logic.
