import os
import csv
import json
import numpy as np
import faiss
import openai

from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------
# OpenAI API Key
# --------------------------------------------------------------------
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual key

# --------------------------------------------------------------------
# Check and Download AllMiniLM-L6-v2 Model
# --------------------------------------------------------------------
MODEL_PATH = "model_repo/all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        print("🔹 Model not found. Downloading AllMiniLM-L6-v2...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODEL_PATH)
        print("✅ Model downloaded and saved at:", MODEL_PATH)
    else:
        print("✅ Model already present at:", MODEL_PATH)

download_model_if_needed()

# Load the model for FAISS embeddings
embedding_model = SentenceTransformer(MODEL_PATH)
EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()

app = Flask(__name__)

# --------------------------------------------------------------------
# GLOBALS for events
# --------------------------------------------------------------------
EVENT_INDEX = None
EVENT_EMBEDDINGS = None
EVENTS_DATA = []

# --------------------------------------------------------------------
# Build FAISS index from CSV
# --------------------------------------------------------------------
def build_faiss_index_for_events(csv_path):
    global EVENT_INDEX, EVENT_EMBEDDINGS, EVENTS_DATA
    events = []
    try:
        with open(csv_path, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(row)
    except Exception as e:
        print("Error loading events CSV:", e)
        return

    if not events:
        print("No events found in CSV.")
        return

    # Convert each row to a single string for embeddings
    texts = [" ".join([row.get("Title", ""), row.get("Description", ""), row.get("Commodity", "")]) for row in events]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    EVENT_INDEX = index
    EVENT_EMBEDDINGS = embeddings
    EVENTS_DATA = events

def faiss_search_events(commodity, top_k=50):
    """Filter events by commodity using FAISS index."""
    if EVENT_INDEX is None:
        print("FAISS index not built.")
        return []
    commodity_emb = embedding_model.encode([commodity], convert_to_numpy=True)
    distances, indices = EVENT_INDEX.search(commodity_emb, top_k)
    return [EVENTS_DATA[idx] for idx in indices[0] if idx < len(EVENTS_DATA)]

# --------------------------------------------------------------------
# GPT-4 Integration for Event Insights
# --------------------------------------------------------------------
def call_gpt4_for_events(events):
    """
    Send the filtered events (in JSON format) to GPT-4 and return the AI insights.
    The prompt instructs GPT-4 to output a JSON array with one object per event,
    each containing: EventID, score, reason, explanation, match_score, region, risks,
    and ai_insights {trends, optimizations}.
    """
    prompt = (
        "I have the following events data in JSON format:\n\n"
        f"{json.dumps(events, indent=2)}\n\n"
        "For each event, provide AI insights as a JSON array, one element per event, "
        "matching the same order and including:\n"
        "  EventID, score (0-100), reason, explanation, match_score (0-1), region, risks, "
        "  ai_insights { trends: [], optimizations: [] }.\n"
        "Output only valid JSON."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in data analysis and event evaluation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        insights_text = response.choices[0].message['content']
        insights = json.loads(insights_text)
        return insights
    except Exception as e:
        print("Error during GPT-4 call (events):", e)
        raise

# --------------------------------------------------------------------
# GPT-4 Integration for Supplier Quotations
# --------------------------------------------------------------------
def call_gpt4_for_suppliers(supplier_text):
    """
    Takes raw supplier quotation text and instructs GPT-4 to parse it into:
      - A 'suppliers' list of objects (rank, supplier_name, price_per_unit, etc.)
      - A 'global_insights' object (trends, risks, optimizations)
    Returns (supplier_list, global_insights).
    """
    prompt = f"""
I have the following supplier quotations data:

{supplier_text}

Please parse the data and provide me a JSON object with two keys:
1) "suppliers": an array of suppliers, each containing:
   - rank (int)
   - supplier_name (string)
   - price_per_unit (string)
   - delivery_date (string)
   - additional_terms (string)
   - score (int from 0 to 100)
   - explanation (string)
   - ai_suggested_percentage (int or float)
2) "global_insights": an object with keys "trends", "risks", and "optimizations", each an array of strings.

Return valid JSON only. Example:
{{
  "suppliers": [
    {{
      "rank": 1,
      "supplier_name": "Supplier A",
      "price_per_unit": "$1000",
      "delivery_date": "2025-01-10",
      "additional_terms": "Fast shipping",
      "score": 95,
      "explanation": "Best price and favorable terms",
      "ai_suggested_percentage": 40
    }},
    ...
  ],
  "global_insights": {{
    "trends": ["..."],
    "risks": ["..."],
    "optimizations": ["..."]
  }}
}}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that parses supplier quotations into structured data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        parsed_text = response.choices[0].message['content']
        parsed_json = json.loads(parsed_text)

        supplier_list = parsed_json.get("suppliers", [])
        global_insights = parsed_json.get("global_insights", {
            "trends": [],
            "risks": [],
            "optimizations": []
        })
        return supplier_list, global_insights
    except Exception as e:
        print("Error during GPT-4 call (suppliers):", e)
        raise

# --------------------------------------------------------------------
#  / Fallback GPT Functions
# --------------------------------------------------------------------
def dummy_gpt4_event_insights(event):
    return {
        "score": np.random.randint(70, 99),
        "reason": " synergy reason",
        "explanation": " chunk-based explanation",
        "match_score": round(np.random.rand(), 2),
        "region": event.get("Region", "Unknown"),
        "risks": "Minimal risk from  data",
        "ai_insights": {
            "trends": [" trend in region."],
            "optimizations": [" optimization approach."]
        }
    }

def dummy_gpt4_supplier_insights():
    """Returns  AI insights for five suppliers, matching your file."""
    return [
        {
            "rank": 1,
            "supplier_name": "Alstom Transport",
            "price_per_unit": "$1,720",
            "delivery_date": "2025-03-01",
            "additional_terms": "Free on-site installation & software calibration",
            "score": 95,
            "explanation": "Most competitive pricing among the set.",
            "ai_suggested_percentage": 30
        },
        {
            "rank": 2,
            "supplier_name": "Siemens Mobility",
            "price_per_unit": "$1,850",
            "delivery_date": "2025-03-10",
            "additional_terms": "Priority manufacturing for urgent orders",
            "score": 90,
            "explanation": "Slightly higher cost but good brand reputation.",
            "ai_suggested_percentage": 25
        },
        {
            "rank": 3,
            "supplier_name": "Schneider Electric Mobility",
            "price_per_unit": "$1,600",
            "delivery_date": "2025-03-05",
            "additional_terms": "Software integration training included",
            "score": 88,
            "explanation": "Lowest cost sensors but mid-range warranty.",
            "ai_suggested_percentage": 20
        },
        {
            "rank": 4,
            "supplier_name": "Hitachi Transportation Systems",
            "price_per_unit": "$1,700",
            "delivery_date": "2025-03-07",
            "additional_terms": "Remote diagnostic support & predictive maintenance",
            "score": 85,
            "explanation": "Competitive pricing, includes AI-based congestion management.",
            "ai_suggested_percentage": 15
        },
        {
            "rank": 5,
            "supplier_name": "ABB Smart Mobility",
            "price_per_unit": "$1,750",
            "delivery_date": "2025-03-12",
            "additional_terms": "Lifetime software updates for sensors",
            "score": 80,
            "explanation": "Slightly higher cost, but strong after-sales service.",
            "ai_suggested_percentage": 10
        }
    ]

# --------------------------------------------------------------------
# Supplier Quotation Loader
# --------------------------------------------------------------------
def load_supplier_quotations(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print("Error loading supplier quotations:", e)
        return ""

# --------------------------------------------------------------------
# Flask Setup
# --------------------------------------------------------------------
@app.before_first_request
def init_app():
    # Use your updated CSV file
    csv_path = os.path.join("data", "Updated_Historical_Event_Data.csv")
    build_faiss_index_for_events(csv_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate_project", methods=["POST"])
def generate_project():
    """
    1) Gather form data
    2) Search events by commodity
    3) Call GPT-4 for event insights
    4) Sort by AI score, show compareEvents.html
    """
    form_data = request.form.to_dict()
    commodity = form_data.get("commodity", "").strip()

    matched_events = faiss_search_events(commodity, top_k=100)
    # If you only want 10 events max:
    matched_events = matched_events[:10]
    if not matched_events:
        return "No events found for the given commodity.", 404

    # GPT-4 for event insights
    try:
        insights = call_gpt4_for_events(matched_events)
        # Build a dict by EventID
        insight_dict = {item.get("EventID"): item for item in insights}
        for evt in matched_events:
            evt_id = evt.get("EventID")
            # Fallback to  if GPT-4 didn't return for this event
            evt["ai_data"] = insight_dict.get(evt_id, dummy_gpt4_event_insights(evt))
    except Exception as e:
        print("Error calling GPT-4 for events:", e)
        # Fallback
        for evt in matched_events:
            evt["ai_data"] = dummy_gpt4_event_insights(evt)

    # Sort by AI score desc
    matched_events.sort(key=lambda x: x["ai_data"]["score"], reverse=True)

    #  global insights
    global_insights = {
        "trends": ["Global demand for sustainable solutions."],
        "risks": ["Potential raw material shortages."],
        "optimizations": ["Leverage multi-year contracts for price stability."]
    }

    return render_template(
        "compareEvents.html",
        events=matched_events,
        global_insights=global_insights,
        form_data=form_data
    )

@app.route("/event_details/<event_id>")
def event_details(event_id):
    """Detailed event page after Add button in compareEvents."""
    event = next((e for e in EVENTS_DATA if e.get("EventID") == event_id), None)
    section_a = "Page 3a: Historical performance..."
    section_b = "Page 3b: Cost analysis / ROI..."
    section_c = "Page 3c: Additional disclaimers..."
    return render_template(
        "event_details.html",
        event=event,
        section_a=section_a,
        section_b=section_b,
        section_c=section_c
    )

@app.route("/quotation_ai/<event_id>")
def quotation_ai(event_id):
    """Intermediary page => leads to Compare Quotes."""
    return render_template("quotation_ai.html", event_id=event_id)

@app.route("/compare_quotes/<event_id>")
def compare_quotes(event_id):
    """
    1) Load raw supplier quotations from a text file (or adapt to your source).
    2) Call GPT-4 to parse them into structured data + global insights.
    3) Render compare_quotes.html with AI insights in a table, plus tabbed insights at the bottom.
    """
    supplier_text = load_supplier_quotations("data/supplier_quotations.txt")  # Update path as needed

    try:
        # Real GPT-4 call
        supplier_insights_list, global_insights = call_gpt4_for_suppliers(supplier_text)
    except Exception as e:
        print("Error calling GPT-4 for suppliers:", e)
        # Fallback
        supplier_insights_list = dummy_gpt4_supplier_insights()
        global_insights = {
            "trends": ["Increasing demand for faster delivery"],
            "risks": ["Price fluctuations in raw materials"],
            "optimizations": ["Consider multi-supplier awarding to mitigate risk"]
        }

    # Sort by 'score' desc
    supplier_insights_list.sort(key=lambda x: x["score"], reverse=True)

    return render_template(
        "compare_quotes.html",
        event_id=event_id,
        supplier_insights_list=supplier_insights_list,
        # Optionally show the raw text if needed (or leave it blank)
        supplier_text="",  
        global_insights=global_insights
    )

@app.route("/award", methods=["POST"])
def award():
    selected_suppliers = request.form.getlist("selected_suppliers")
    return render_template("award_result.html", awarded_suppliers=selected_suppliers)

if __name__ == "__main__":
    app.run(debug=True)
