# agenttest
To create conversational ai
<!DOCTYPE html>
<html>
<head>
  <title>Compare Events - AI Insights</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css">
  <style>
    .card-header h5 {
      font-weight: 600;
    }
  </style>
  <script>
    function toggleInsights(uniqueId) {
      var block = document.getElementById('insights-' + uniqueId);
      var arrowBtn = document.getElementById('arrow-btn-' + uniqueId);
      if (block.style.display === 'none') {
         block.style.display = 'block';
         arrowBtn.innerHTML = '▲';
      } else {
         block.style.display = 'none';
         arrowBtn.innerHTML = '▼';
      }
    }
  </script>
</head>
<body>
  <!-- Navbar with SAP Logo -->
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary shadow">
    <div class="container-fluid">
      <a class="navbar-brand" href="{{ url_for('index') }}">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/SAP_2011_logo.svg/150px-SAP_2011_logo.svg.png" 
             alt="SAP Logo" height="40">
        BidGenius
      </a>
    </div>
  </nav>

  <div class="container mt-4">
      <h2 class="mb-4">Similar Sourcing Events by BidGenius</h2>
      <div class="mb-3">
        <p><strong>Description:</strong> {{ form_data.get('description','') }}</p>
        <p><strong>Project Type:</strong> {{ form_data.get('project_type','Single Event') }}</p>
        <p><strong>Commodity:</strong> {{ form_data.get('commodity','') }}</p>
      </div>

      <hr>
      <h3 class="mb-3">AI Insights for Events</h3>
      {% for event in events %}
      <div class="card mb-3 shadow-sm">
          <div class="card-header d-flex justify-content-between align-items-center bg-light">
              <h5 class="mb-0">{{ event.Title if event.Title else event.EventID }}</h5>
              <!-- Unique ID to avoid conflicts -->
              <button id="arrow-btn-{{ loop.index }}-{{ event.EventID }}" 
                      class="btn btn-sm btn-info"
                      onclick="toggleInsights('{{ loop.index }}-{{ event.EventID }}')"
                      style="width: 40px;">
                ▼
              </button>
          </div>
          <div class="card-body">
              <div id="insights-{{ loop.index }}-{{ event.EventID }}" style="display: none;">
                  {% if event.ai_data %}
                  <p><strong>Score:</strong> {{ event.ai_data.score }}</p>
                  <p><strong>Reason:</strong> {{ event.ai_data.reason }}</p>
                  <p><strong>Explanation:</strong> {{ event.ai_data.explanation }}</p>
                  <p><strong>Match Score:</strong> {{ event.ai_data.match_score }}</p>
                  <!-- Show region from AI data if available; otherwise from event data -->
                  <p><strong>Region:</strong> {{ event.ai_data.region if event.ai_data.region|default("") != "" else event.get("Region", "Unknown") }}</p>
                  <p><strong>Risks:</strong> {{ event.ai_data.risks }}</p>
                  <hr>
                  <h6>Extended AI Insights</h6>
                  <p><strong>Trends:</strong> {{ event.ai_data.ai_insights.trends|join(', ') }}</p>
                  <p><strong>Optimizations:</strong> {{ event.ai_data.ai_insights.optimizations|join(', ') }}</p>
                  {% else %}
                  <p>No AI insights available for this event.</p>
                  {% endif %}
              </div>
              <a href="{{ url_for('event_details', event_id=event.EventID) }}" class="btn btn-success mt-2">Add</a>
          </div>
      </div>
      {% endfor %}
      <hr>

      <h4>AI (Global) Insights</h4>
      <ul class="nav nav-tabs" id="globalInsightsTab" role="tablist">
          <li class="nav-item" role="presentation">
             <button class="nav-link active" id="trends-tab" data-bs-toggle="tab" data-bs-target="#globalTrends" type="button" role="tab">Trends</button>
          </li>
          <li class="nav-item" role="presentation">
             <button class="nav-link" id="risks-tab" data-bs-toggle="tab" data-bs-target="#globalRisks" type="button" role="tab">Risks</button>
          </li>
          <li class="nav-item" role="presentation">
             <button class="nav-link" id="optimizations-tab" data-bs-toggle="tab" data-bs-target="#globalOptimizations" type="button" role="tab">Optimizations</button>
          </li>
      </ul>
      <div class="tab-content mt-3">
          <div id="globalTrends" class="tab-pane fade show active" role="tabpanel">
             <p>{{ global_insights.trends|join(', ') }}</p>
          </div>
          <div id="globalRisks" class="tab-pane fade" role="tabpanel">
             <p>{{ global_insights.risks|join(', ') }}</p>
          </div>
          <div id="globalOptimizations" class="tab-pane fade" role="tabpanel">
             <p>{{ global_insights.optimizations|join(', ') }}</p>
          </div>
      </div>
  </div>

  <!-- Footer -->
  <footer class="bg-dark text-center text-white py-3 mt-5">
      <small>&copy; 2025 BidGenius. All rights reserved.</small>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
