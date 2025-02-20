# agenttest
To create conversational ai

<!DOCTYPE html>
<html>
<head>
  <title>Event Details</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css">
</head>
<body>
  <!-- Example Navbar -->
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
      <h2>{{ event.Title if event.Title else event.EventID }}</h2>
      <p>This is your Event Details page. You can show minimal info here or any disclaimers.</p>
      <!-- Button to go to the new "viewQuotes" page -->
      <a href="{{ url_for('view_quotes', event_id=event.EventID) }}" class="btn btn-success">View Quotes</a>
  </div>

  <!-- Footer -->
  <footer class="bg-dark text-center text-white py-3 mt-5">
      <small>&copy; 2025 BidGenius. All rights reserved.</small>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
