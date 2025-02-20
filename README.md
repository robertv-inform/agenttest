# agenttest
To create conversational ai

<!DOCTYPE html>
<html>
<head>
  <title>View Quotes</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css">
</head>
<body>
  <!-- Navbar (optional) -->
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
      <h2>Supplier Quotations for Event {{ event_id }}</h2>
      <p>Below is the raw data from <strong>supplier_quotations.txt</strong>:</p>

      <!-- Show the raw text in a <pre> block for readability -->
      <div class="mb-3">
        <pre style="max-height: 400px; overflow:auto;">{{ raw_text }}</pre>
      </div>

      <!-- Button to go to the GPT-4 comparison page -->
      <a href="{{ url_for('compare_quotes', event_id=event_id) }}" class="btn btn-success">
        Compare Quotes with GPT-4
      </a>
  </div>

  <!-- Footer -->
  <footer class="bg-dark text-center text-white py-3 mt-5">
      <small>&copy; 2025 BidGenius. All rights reserved.</small>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
