# agenttest
To create conversational ai

"""
    Takes a JSON array of events and instructs GPT-4 to parse it into event insights.
    Each event insight object must contain exactly the following keys:
      - "EventID": the event's identifier,
      - "score": an integer from 0 to 100,
      - "reason": a short explanation for the score,
      - "explanation": a detailed explanation of the analysis,
      - "match_score": a float between 0 and 1,
      - "region": the region associated with the event,
      - "risks": a string describing potential risks,
      - "ai_insights": an object with keys "trends" and "optimizations", each an array of strings.
    Return only valid JSON.
    """
    prompt = f"""
I have the following events data in JSON format:

{json.dumps(events, indent=2)}

Please parse this data and provide me a JSON array where each element is an object corresponding to an event's AI insights. Each object must contain exactly the following keys:
- "EventID": the event's identifier,
- "score": an integer from 0 to 100,
- "reason": a short explanation for the score,
- "explanation": a detailed explanation of the analysis,
- "match_score": a float between 0 and 1,
- "region": the region associated with the event,
- "risks": a string describing potential risks,
- "ai_insights": an object with two keys:
    - "trends": an array of strings representing trends,
    - "optimizations": an array of strings representing optimization suggestions.

Return only valid JSON.
"""
