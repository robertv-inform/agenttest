# agenttest
To create conversational ai
@app.route("/view_quotes/<event_id>")
def view_quotes(event_id):
    """
    A page to display the raw supplier quotations text for a given event,
    and optionally link to the GPT-4 comparison page.
    """
    # Load the raw text
    raw_text = load_supplier_quotations("data/supplier_quotations.txt")
    # You can also load an event object if needed
    event = next((e for e in EVENTS_DATA if e.get("EventID") == event_id), None)
    
    return render_template("viewQuotes.html", event=event, event_id=event_id, raw_text=raw_text)