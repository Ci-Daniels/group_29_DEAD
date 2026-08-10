// verify.js
// Handles the verification flow: submits a beneficiary ID, starts a
// verification session, drives the live preview + status panel, and shows
// a modal with the final VERIFIED / NOT VERIFIED result.

const form = document.getElementById("verify-form");
const startBtn = document.getElementById("start-btn");
const videoFeed = document.getElementById("video-feed");
const videoPlaceholder = document.getElementById("video-placeholder");
const statusText = document.getElementById("status-text");
const progressText = document.getElementById("progress-text");

const modal = document.getElementById("result-modal");
const modalIcon = document.getElementById("modal-icon");
const modalTitle = document.getElementById("modal-title");
const modalSubtitle = document.getElementById("modal-subtitle");
const modalScore = document.getElementById("modal-score");
const modalCloseBtn = document.getElementById("modal-close-btn");
const modalRetryBtn = document.getElementById("modal-retry-btn");

let pollTimer = null;

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

/** Show the result modal in either success or error styling. */
function showModal({ success, title, subtitle, score }) {
  modal.classList.toggle("modal--success", success);
  modal.classList.toggle("modal--error", !success);
  modalIcon.textContent = success ? "✔" : "✖";
  modalTitle.textContent = title;
  modalSubtitle.textContent = subtitle;
  modalScore.textContent = score || "";
  modal.hidden = false;
}

/** Poll /api/session/<id>/status until the verification session finishes. */
function pollStatus(sessionId) {
  pollTimer = setInterval(async () => {
    const res = await fetch(`/api/session/${sessionId}/status`);
    const data = await res.json();

    setStatus(data.status, data.progress);

    if (data.done) {
      stopPolling();
      startBtn.disabled = false;

      if (data.error) {
        showModal({
          success: false,
          title: "Verification Failed",
          subtitle: data.error,
        });
        return;
      }

      const result = data.result;
      if (result && result.verified) {
        showModal({
          success: true,
          title: "Identity Verified",
          subtitle: "Access Granted",
          score: `Similarity score: ${result.similarity_score.toFixed(4)}`,
        });
      } else {
        showModal({
          success: false,
          title: "Verification Failed",
          subtitle: "Access Denied",
          score: result ? `Similarity score: ${result.similarity_score.toFixed(4)}` : "",
        });
      }
    }
  }, 700);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  stopPolling();              // kill any interval left over from a previous run
  modal.hidden = true;
  modal.classList.remove("modal--success", "modal--error");
  startBtn.disabled = true;

  // reset preview back to "not started" so a stale frame doesn't linger
  videoFeed.hidden = true;
  videoFeed.removeAttribute("src");
  videoPlaceholder.hidden = false;

  const payload = Object.fromEntries(new FormData(form).entries());

  const res = await fetch("/api/verify/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  if (data.error) {
    setStatus(data.error, "");
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
});

modalCloseBtn.addEventListener("click", () => {
  modal.hidden = true;
  window.location.href = "/";
});

modalRetryBtn.addEventListener("click", () => {
  modal.hidden = true;
});
