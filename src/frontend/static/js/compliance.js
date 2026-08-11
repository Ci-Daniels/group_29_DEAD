// compliance.js
// Dummy handler for document upload form.

(function () {
  var form = document.getElementById("compliance-form");
  var msgEl = document.getElementById("compliance-message");

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    msgEl.textContent = "Document uploaded successfully! Pending review.";
    msgEl.hidden = false;
    setTimeout(function () { msgEl.hidden = true; }, 2500);
    form.reset();
  });
})();