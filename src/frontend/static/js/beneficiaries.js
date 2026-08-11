// beneficiaries.js
// Manages next of kin CRUD with dummy data. Replace with real API calls later.

(function () {
  var dummyBeneficiaries = [
    { id: 1, name: "Jane Doe", national_id: "B87654321", relationship: "Spouse", role: "admin", decision_status: "Accepted", face_enrolled: true },
    { id: 2, name: "Michael Doe", national_id: "C11223344", relationship: "Child", role: "viewer", decision_status: "Pending", face_enrolled: false },
    { id: 3, name: "Sarah Wanjiku", national_id: "D99887766", relationship: "Sibling", role: "executor", decision_status: "Accepted", face_enrolled: true }
  ];

  var form = document.getElementById("beneficiary-form");
  var tbody = document.getElementById("beneficiary-tbody");
  var emptyState = document.getElementById("beneficiary-empty");
  var countEl = document.getElementById("beneficiary-count");
  var msgEl = document.getElementById("beneficiary-message");

  var nextId = 4;

  function getStatusBadge(status) {
    if (status === "Accepted") return '<span class="badge badge--success">Accepted</span>';
    if (status === "Pending") return '<span class="badge badge--warning">Pending</span>';
    if (status === "Declined") return '<span class="badge badge--error">Declined</span>';
    return '<span class="badge badge--neutral">' + status + '</span>';
  }

  function render() {
    if (dummyBeneficiaries.length === 0) {
      emptyState.hidden = false;
      tbody.innerHTML = "";
    } else {
      emptyState.hidden = true;
      tbody.innerHTML = dummyBeneficiaries.map(function (b) {
        return "<tr>" +
          "<td>" + b.name + "</td>" +
          "<td>" + b.national_id + "</td>" +
          "<td>" + b.relationship + "</td>" +
          "<td>" + b.role.charAt(0).toUpperCase() + b.role.slice(1) + "</td>" +
          "<td>" + getStatusBadge(b.decision_status) + "</td>" +
          "<td>" + (b.face_enrolled ? '<span class="badge badge--success">Enrolled</span>' : '<span class="badge badge--neutral">Not Enrolled</span>') + "</td>" +
          "<td><button class='btn btn--ghost btn--sm' data-delete='" + b.id + "'>Remove</button></td>" +
          "</tr>";
      }).join("");
    }
    countEl.textContent = dummyBeneficiaries.length + " beneficiar" + (dummyBeneficiaries.length !== 1 ? "ies" : "y");
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (!form.checkValidity()) { form.reportValidity(); return; }

    var fd = new FormData(form);
    dummyBeneficiaries.push({
      id: nextId++,
      name: fd.get("name"),
      national_id: fd.get("national_id"),
      relationship: fd.get("relationship"),
      role: fd.get("role"),
      decision_status: "Pending",
      face_enrolled: false
    });
    render();
    form.reset();
    msgEl.textContent = "Beneficiary added successfully!";
    msgEl.hidden = false;
    setTimeout(function () { msgEl.hidden = true; }, 2500);
  });

  tbody.addEventListener("click", function (e) {
    if (e.target.hasAttribute("data-delete")) {
      var id = parseInt(e.target.getAttribute("data-delete"), 10);
      dummyBeneficiaries = dummyBeneficiaries.filter(function (b) { return b.id !== id; });
      render();
    }
  });

  render();
})();