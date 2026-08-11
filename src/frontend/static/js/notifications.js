// notifications.js
// Renders notification list from dummy data.

(function () {
  var dummyNotifications = [
    { id: 1, status: "Sent", message: "Death certificate verified for primary holder John Doe. Initiating beneficiary notification.", date_sent: "2025-08-10", recipient: "Jane Doe (Spouse)" },
    { id: 2, status: "Pending", message: "Awaiting death certificate submission from legal authority.", date_sent: "2025-08-01", recipient: "System" },
    { id: 3, status: "Read", message: "Beneficiary Jane Doe accepted inheritance terms. Proceeding to compliance.", date_sent: "2025-08-11", recipient: "John Doe (Primary)" },
    { id: 4, status: "Sent", message: "Reminder: Michael Doe has not responded to inheritance notification.", date_sent: "2025-08-09", recipient: "Michael Doe (Child)" },
    { id: 5, status: "Failed", message: "Email delivery to sarah@example.com failed. Please update contact information.", date_sent: "2025-07-28", recipient: "Sarah Wanjiku (Sibling)" }
  ];

  var tbody = document.getElementById("notification-tbody");
  var emptyState = document.getElementById("notification-empty");

  function getStatusBadge(status) {
    if (status === "Sent") return '<span class="badge badge--neutral">Sent</span>';
    if (status === "Read") return '<span class="badge badge--success">Read</span>';
    if (status === "Pending") return '<span class="badge badge--warning">Pending</span>';
    if (status === "Failed") return '<span class="badge badge--error">Failed</span>';
    return '<span class="badge badge--neutral">' + status + '</span>';
  }

  function render() {
    if (dummyNotifications.length === 0) {
      emptyState.hidden = false;
      tbody.innerHTML = "";
    } else {
      emptyState.hidden = true;
      tbody.innerHTML = dummyNotifications.map(function (n) {
        return "<tr>" +
          "<td>" + getStatusBadge(n.status) + "</td>" +
          "<td>" + n.message + "</td>" +
          "<td>" + n.date_sent + "</td>" +
          "<td>" + n.recipient + "</td>" +
          "</tr>";
      }).join("");
    }
  }

  render();
})();