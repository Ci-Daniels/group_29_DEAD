// failsafe.js
// Dummy handler for fail-safe settings form.

(function () {
  var form = document.getElementById("failsafe-form");
  var msgEl = document.getElementById("failsafe-message");

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    msgEl.textContent = "Fail-safe settings saved successfully!";
    msgEl.hidden = false;
    setTimeout(function () { msgEl.hidden = true; }, 2500);
  });
})();