// register.js
// Handles the registration form: collects beneficiary info, gates on a
// biometric-data consent modal, then starts an enrollment session on the
// Flask backend and drives the live preview + status panel while
// SCRFD/ArcFace do their work server-side.

const form = document.getElementById("register-form");
const startBtn = document.getElementById("start-btn");
const videoFeed = document.getElementById("video-feed");
const videoPlaceholder = document.getElementById("video-placeholder");
const statusText = document.getElementById("status-text");
const progressText = document.getElementById("progress-text");
const resultBanner = document.getElementById("result-banner");

// -- consent modal elements --------------------------------------------
const consentModal = document.getElementById("consent-modal");
const consentNamePlaceholder = document.getElementById("consent-name-placeholder");
const consentCheckbox = document.getElementById("consent-checkbox");
const consentAgreeBtn = document.getElementById("consent-agree-btn");
const consentDeclineBtn = document.getElementById("consent-decline-btn");

let pollTimer = null;
let pendingPayload = null; // form data waiting on consent before it's sent

/** Update the status strip text. */
function setStatus(text, progress) {
  statusText.textContent = text || "Waiting to start...";
  progressText.textContent = progress || "";
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

/** Poll /api/session/<id>/status until the enrollment session finishes. */
function pollStatus(sessionId) {
  pollTimer = setInterval(async () => {
    const res = await fetch(`/api/session/${sessionId}/status`);
    const data = await res.json();

    setStatus(data.status, data.progress);

    if (data.done) {
      stopPolling();
      startBtn.disabled = false;

      if (data.error) {
        resultBanner.className = "result-banner result-banner--error";
        resultBanner.textContent = `Enrollment failed: ${data.error}`;
      } else {
        resultBanner.className = "result-banner result-banner--success";
        resultBanner.textContent = "Beneficiary enrolled successfully.";
      }
      resultBanner.hidden = false;
    }
  }, 700);
}

/** Actually starts enrollment on the backend. Same logic as before -- only
 *  now called after consent has been confirmed, instead of directly from
 *  the form submit handler. */
async function startEnrollment(payload) {
  stopPolling(); // kill any interval left over from a previous run
  resultBanner.hidden = true;
  startBtn.disabled = true;

  videoFeed.hidden = true;
  videoFeed.removeAttribute("src");
  videoPlaceholder.hidden = false;

  const res = await fetch("/api/register/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  if (data.error) {
    resultBanner.className = "result-banner result-banner--error";
    resultBanner.textContent = data.error;
    resultBanner.hidden = false;
    startBtn.disabled = false;
    return;
  }

  // Point the <img> tag at the MJPEG stream for this session.
  videoPlaceholder.hidden = true;
  videoFeed.hidden = false;
  videoFeed.src = `/video_feed/${data.session_id}`;
  
    // Match the frame box to the REAL camera resolution once the stream
  // actually loads -- eliminates letterbox bars without hardcoding a
  // fixed aspect-ratio that drifts out of sync whenever the camera
  // negotiates a different size.
  videoFeed.onload = () => {
    if (videoFeed.naturalWidth && videoFeed.naturalHeight) {
      videoFeed.parentElement.style.aspectRatio = `${videoFeed.naturalWidth} / ${videoFeed.naturalHeight}`;
    }
  };

  setStatus("Starting camera...", "");
  pollStatus(data.session_id);
}

// -- consent gate ---------------------------------------------------------
function openConsentModal(payload) {
  pendingPayload = payload;
  consentNamePlaceholder.textContent = payload.name || "this beneficiary";
  consentCheckbox.checked = false;
  consentAgreeBtn.disabled = true;
  consentModal.hidden = false;
}

function closeConsentModal() {
  consentModal.hidden = true;
}

consentCheckbox.addEventListener("change", () => {
  consentAgreeBtn.disabled = !consentCheckbox.checked;
});

consentAgreeBtn.addEventListener("click", () => {
  if (!pendingPayload) return;

  // Record consent alongside the rest of the beneficiary metadata.
  pendingPayload.consent_given = true;
  pendingPayload.consent_timestamp = new Date().toISOString();

  closeConsentModal();
  startEnrollment(pendingPayload);
  pendingPayload = null;
});

consentDeclineBtn.addEventListener("click", () => {
  pendingPayload = null;
  closeConsentModal();
});

// -- form submit now opens the consent gate instead of starting directly --
form.addEventListener("submit", (event) => {
  event.preventDefault();

  // Let native HTML validation (required fields) run first.
  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  const payload = Object.fromEntries(new FormData(form).entries());
  openConsentModal(payload);
});