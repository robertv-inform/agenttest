# agenttest
To create conversational ai
<!DOCTYPE html>
<html>
<head>
  <title>Compare Quotes - Event {{ event_id }}</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css">
  <style>
    .table th {
      background-color: #f2f2f2;
    }
    .table td, .table th {
      vertical-align: middle;
    }
  </style>
</head>
<body>
  <!-- Navbar with SAP Logo (external) -->
  <nav class="navbar navbar-expand-lg navbar-light bg-light">
    <div class="container-fluid">
      <a class="navbar-brand" href="{{ url_for('index') }}">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/SAP_2011_logo.svg/200px-SAP_2011_logo.svg.png" alt="SAP Logo" width="120">
      </a>
    </div>
  </nav>

  <div class="container mt-4">
      <h2>Quote Comparison for Event {{ event_id }}</h2>
      <form action="{{ url_for('award') }}" method="post">
          <table class="table table-bordered table-striped align-middle">
              <thead>
                  <tr>
                      <th>Option</th>
                      <th>Rank</th>
                      <th>Supplier Name</th>
                      <th>Price Per Unit</th>
                      <th>Delivery Date</th>
                      <th>Additional Terms</th>
                      <th>Score</th>
                      <th>Explanation</th>
                      <th>AI Suggested %</th>
                      <th>Select</th>
                  </tr>
              </thead>
              <tbody>
              {% for sup in supplier_insights_list %}
                  <tr>
                      <td>Quote {{ loop.index }}</td>
                      <td>{{ sup.rank }}</td>
                      <td>{{ sup.supplier_name }}</td>
                      <td>{{ sup.price_per_unit }}</td>
                      <td>{{ sup.delivery_date }}</td>
                      <td>{{ sup.additional_terms }}</td>
                      <td>{{ sup.score }}</td>
                      <td>{{ sup.explanation }}</td>
                      <td style="max-width: 120px;">
                          <input type="number" class="form-control form-control-sm"
                                 name="percentage_{{ sup.supplier_name }}"
                                 value="{{ sup.ai_suggested_percentage }}">
                      </td>
                      <td class="text-center">
                          <input type="checkbox" name="selected_suppliers" value="{{ sup.supplier_name }}">
                      </td>
                  </tr>
              {% endfor %}
              </tbody>
          </table>
          <button type="submit" class="btn btn-primary mt-2">Award</button>
      </form>

      <!-- AI Insights Tabs -->
      <div class="mt-5">
          <h4>AI Insights</h4>
          <ul class="nav nav-tabs">
              <li class="nav-item">
                  <a class="nav-link active" data-bs-toggle="tab" href="#trends">Trends</a>
              </li>
              <li class="nav-item">
                  <a class="nav-link" data-bs-toggle="tab" href="#risks">Risks</a>
              </li>
              <li class="nav-item">
                  <a class="nav-link" data-bs-toggle="tab" href="#optimizations">Optimizations</a>
              </li>
          </ul>
          <div class="tab-content border border-top-0 p-3">
              <div id="trends" class="container tab-pane active">
                  <p>{{ global_insights.trends | join(', ') }}</p>
              </div>
              <div id="risks" class="container tab-pane fade">
                  <p>{{ global_insights.risks | join(', ') }}</p>
              </div>
              <div id="optimizations" class="container tab-pane fade">
                  <p>{{ global_insights.optimizations | join(', ') }}</p>
              </div>
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
