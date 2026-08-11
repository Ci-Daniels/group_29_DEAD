// signup.js
// Handles signup form submission. Currently uses dummy validation.
// Replace fetch URL and logic when backend auth API is ready.

const form = document.getElementById("signup-form");
const errorEl = document.getElementById("signup-error");
const successEl = document.getElementById("signup-success");

form.addEventListener("submit", function (event) {
  event.preventDefault();

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = Object.fromEntries(new FormData(form).entries());
  errorEl.hidden = true;
  successEl.hidden = true;

  // Client-side password match check
  if (payload.password !== payload.confirm_password) {
    errorEl.textContent = "Passwords do not match.";
    errorEl.hidden = false;
    return;
  }

  // Dummy signup -- replace with real API call
  successEl.textContent = "Account created successfully! Redirecting to login...";
  successEl.hidden = false;

  setTimeout(function () {
    window.location.href = "/login";
  }, 2000);
});