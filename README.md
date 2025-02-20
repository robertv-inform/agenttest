# agenttest
To create conversational ai

@app.route("/award", methods=["POST"])
def award():
    selected_suppliers = request.form.getlist("selected_suppliers")
    percentages = {}
    for key, value in request.form.items():
        if key.startswith("percentage_"):
            percentages[key] = value
    return render_template("award_result.html", awarded_suppliers=selected_suppliers, percentages=percentages)
