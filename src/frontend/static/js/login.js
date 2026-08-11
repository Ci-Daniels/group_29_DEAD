// login.js
// Handles login form submission. Currently uses dummy validation.
// Replace fetch URL and logic when backend auth API is ready.

const form = document.getElementById("login-form");
const errorEl = document.getElementById("login-error");

form.addEventListener("submit", function (event) {
  event.preventDefault();

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = Object.fromEntries(new FormData(form).entries());
  errorEl.hidden = true;

  // Dummy validation -- replace with real API call
  if (payload.email === "user@dead.com" && payload.password === "password") {
    window.location.href = "/dashboard";
  } else {
    errorEl.textContent = "Invalid email or password. (Hint: use user@dead.com / password)";
    errorEl.hidden = false;
  }
});