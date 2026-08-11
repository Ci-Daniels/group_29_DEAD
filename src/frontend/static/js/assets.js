// assets.js
// Manages asset CRUD with dummy data. Replace with real API calls later.

(function () {
  var dummyAssets = [
    { id: 1, asset_type: "Bank Account", asset_location: "Equity Bank Kenya", asset_value: 12500, date_created: "2025-06-15" },
    { id: 2, asset_type: "Mobile Money", asset_location: "M-Pesa", asset_value: 3200, date_created: "2025-06-14" },
    { id: 3, asset_type: "Crypto Wallet", asset_location: "Binance", asset_value: 8750, date_created: "2025-06-10" },
    { id: 4, asset_type: "Social Media (Influencer)", asset_location: "YouTube", asset_value: 5000, date_created: "2025-05-28" }
  ];

  var form = document.getElementById("asset-form");
  var tbody = document.getElementById("asset-tbody");
  var emptyState = document.getElementById("asset-empty");
  var countEl = document.getElementById("asset-count");
  var msgEl = document.getElementById("asset-message");

  var nextId = 5;

  function formatCurrency(val) {
    return "$" + Number(val).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function render() {
    if (dummyAssets.length === 0) {
      emptyState.hidden = false;
      tbody.innerHTML = "";
    } else {
      emptyState.hidden = true;
      tbody.innerHTML = dummyAssets.map(function (a) {
        return "<tr>" +
          "<td>" + a.asset_type + "</td>" +
          "<td>" + a.asset_location + "</td>" +
          "<td>" + formatCurrency(a.asset_value) + "</td>" +
          "<td>" + a.date_created + "</td>" +
          "<td><button class='btn btn--ghost btn--sm' data-delete='" + a.id + "'>Delete</button></td>" +
          "</tr>";
      }).join("");
    }
    countEl.textContent = dummyAssets.length + " asset" + (dummyAssets.length !== 1 ? "s" : "");
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (!form.checkValidity()) { form.reportValidity(); return; }

    var fd = new FormData(form);
    dummyAssets.push({
      id: nextId++,
      asset_type: fd.get("asset_type"),
      asset_location: fd.get("asset_location"),
      asset_value: parseFloat(fd.get("asset_value")),
      date_created: fd.get("date_created")
    });
    render();
    form.reset();
    msgEl.textContent = "Asset added successfully!";
    msgEl.hidden = false;
    setTimeout(function () { msgEl.hidden = true; }, 2500);
  });

  tbody.addEventListener("click", function (e) {
    if (e.target.hasAttribute("data-delete")) {
      var id = parseInt(e.target.getAttribute("data-delete"), 10);
      dummyAssets = dummyAssets.filter(function (a) { return a.id !== id; });
      render();
    }
  });

  render();
})();