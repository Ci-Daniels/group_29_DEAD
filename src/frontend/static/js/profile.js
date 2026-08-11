// profile.js
// Handles profile update and consent toggle saving.

(function () {
  var profileForm = document.getElementById("profile-form");
  var profileMsg = document.getElementById("profile-message");

  profileForm.addEventListener("submit", function (e) {
    e.preventDefault();
    profileMsg.textContent = "Profile updated successfully!";
    profileMsg.hidden = false;
    setTimeout(function () { profileMsg.hidden = true; }, 2500);
  });

  var saveConsentBtn = document.getElementById("save-consent-btn");
  var consentMsg = document.getElementById("consent-message");

  saveConsentBtn.addEventListener("click", function () {
    consentMsg.textContent = "Consent preferences saved.";
    consentMsg.hidden = false;
    setTimeout(function () { consentMsg.hidden = true; }, 2500);
  });
})();