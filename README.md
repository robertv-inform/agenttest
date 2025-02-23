# agenttest
To create conversational ai

<!DOCTYPE html>
<html>
<head>
  <title>Create Sourcing Project - BidGenius</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css">
  <style>
    /* Hero section styling for an enterprise look */
    .hero {
      background: #f8f9fa url("https://via.placeholder.com/1600x400/cccccc/ffffff?text=Tender+Background") no-repeat center center;
      background-size: cover;
      padding: 80px 0;
      margin-bottom: 30px;
    }
    .hero-text {
      background-color: rgba(255, 255, 255, 0.9);
      padding: 20px 30px;
      border-radius: 10px;
      display: inline-block;
    }
  </style>
</head>
<body>
  <!-- Navbar with SAP Logo -->
  <nav class="navbar navbar-expand-lg navbar-dark bg-primary shadow">
    <div class="container-fluid">
      <a class="navbar-brand" href="#">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/SAP_2011_logo.svg/150px-SAP_2011_logo.svg.png" 
             alt="SAP Logo" height="40">
        BidGenius
      </a>
    </div>
  </nav>

  <!-- Hero Section -->
  <div class="hero text-center">
    <div class="hero-text">
      <h1 class="display-5 fw-bold">Create Sourcing Project</h1>
      <p class="lead">Kick off your tender project and let AI power your insights.</p>
    </div>
  </div>

  <!-- Form Section -->
  <div class="container mb-5">
    <h2 class="mb-4">Enter Tender Details</h2>
    <!-- This form sends data to /generate_project (your existing backend) -->
    <form id="sourcingForm" action="{{ url_for('generate_project') }}" method="post" class="bg-light p-4 rounded shadow">
      <div class="mb-3">
        <label for="project_name" class="form-label fw-semibold">Project Name</label>
        <input type="text" class="form-control" id="project_name" name="project_name" placeholder="Enter project name">
      </div>
      <div class="mb-3">
        <label for="description" class="form-label fw-semibold">Project Description</label>
        <input type="text" class="form-control" id="description" name="description" placeholder="Enter project description">
      </div>
      <div class="row mb-3">
        <div class="col-md-4">
          <label for="commodity" class="form-label fw-semibold">Commodity</label>
          <input type="text" class="form-control" id="commodity" name="commodity" placeholder="e.g., Electronics">
        </div>
        <div class="col-md-4">
          <label for="contract_effective_date" class="form-label fw-semibold">Contract Effective Date</label>
          <input type="date" class="form-control" id="contract_effective_date" name="contract_effective_date">
        </div>
        <div class="col-md-4">
          <label for="currency" class="form-label fw-semibold">Currency</label>
          <input type="text" class="form-control" id="currency" name="currency" placeholder="e.g., USD">
        </div>
      </div>
      <!-- (Include additional fields as needed) -->
      <button type="button" class="btn btn-primary px-4 py-2" onclick="openInNewWindow()">
        Generate Similar Sourcing Project
      </button>
    </form>
  </div>

  <!-- Footer -->
  <footer class="bg-dark text-center text-white py-3">
    <small>&copy; 2025 BidGenius. All rights reserved.</small>
  </footer>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/js/bootstrap.bundle.min.js"></script>
  <script>
    function openInNewWindow() {
      var form = document.getElementById("sourcingForm");
      // Set the form target to a new window (named "newWindow")
      form.target = "newWindow";
      // Open a new window with desired dimensions (adjust as needed)
      window.open("", "newWindow", "width=900,height=700");
      // Submit the form; the backend processes the data and returns compareEvents.html
      form.submit();
    }
  </script>
</body>
</html>
