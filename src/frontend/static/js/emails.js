const scanTrigger = document.getElementById("scan-trigger");
const progressWrap = document.getElementById("scan-progress");
const scanStatus = document.getElementById("scan-status");
const scanPercent = document.getElementById("scan-percent");
const scanFill = document.getElementById("scan-fill");
const scanError = document.getElementById("scan-error");
const findingsContainer = document.getElementById("findings-container");
const emptyFindings = document.getElementById("empty-findings");

let activePollTimer = null;
const renderedFindingIds = new Set();

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setProgress(progress, statusText) {
  const safeProgress = Math.max(0, Math.min(100, Number(progress) || 0));
  progressWrap.style.display = "grid";
  scanFill.style.width = `${safeProgress}%`;
  scanPercent.textContent = `${safeProgress}%`;
  scanStatus.textContent = statusText || "Scanning inbox...";
}

function clearResults() {
  findingsContainer.innerHTML = "";
  emptyFindings.style.display = "none";
  renderedFindingIds.clear();
}

function findingUniqueId(finding, indexHint = 0) {
  return (
    finding.message_id ||
    `${finding.subject || ""}|${finding.date || ""}|${finding.asset_provider || ""}|${indexHint}`
  );
}

function findingCardHtml(finding) {
  const subject = escapeHtml(finding.subject || "(No Subject)");
  const provider = escapeHtml(finding.asset_provider || "Unknown provider");
  const date = escapeHtml(finding.date || "Unknown date");
  const confidence = escapeHtml(finding.surety_percentage ?? "N/A");
  const category = escapeHtml(finding.category || "uncategorized");
  const reasoning = escapeHtml(finding.reasoning || "No reasoning available.");

  return `
    <article class="email-item">
      <h2 class="email-item__subject">${subject}</h2>
      <p class="email-item__meta">
        Provider: ${provider}<br>
        Date: ${date}<br>
        Confidence: ${confidence}%
      </p>
      <span class="badge">${category}</span>
      <p class="email-item__snippet">${reasoning}</p>
    </article>
  `;
}

function appendFindings(findings) {
  if (!Array.isArray(findings) || findings.length === 0) {
    return;
  }

  const newCards = [];

  findings.forEach((finding, index) => {
    const uid = findingUniqueId(finding, index);
    if (renderedFindingIds.has(uid)) {
      return;
    }
    renderedFindingIds.add(uid);
    newCards.push(findingCardHtml(finding));
  });

  if (newCards.length > 0) {
    emptyFindings.style.display = "none";
    findingsContainer.insertAdjacentHTML("beforeend", newCards.join(""));
  }
}

function renderFindings(findings) {
  clearResults();

  if (!Array.isArray(findings) || findings.length === 0) {
    emptyFindings.style.display = "block";
    return;
  }

  appendFindings(findings);
}

function showError(message) {
  scanError.textContent = message || "Unexpected error while scanning emails.";
  scanError.style.display = "block";
}

function clearError() {
  scanError.textContent = "";
  scanError.style.display = "none";
}

async function pollScanStatus(sessionId) {
  if (activePollTimer) {
    clearTimeout(activePollTimer);
  }

  try {
    const response = await fetch(`/api/email-scan/${encodeURIComponent(sessionId)}/status`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Unable to fetch scanning progress.");
    }

    setProgress(data.progress || 0, data.status || "Scanning inbox...");

    appendFindings(data.partial_findings || []);

    if (data.done) {
      scanTrigger.disabled = false;

      if (data.error) {
        showError(data.error);
        return;
      }

      setProgress(100, "Scan complete.");
      renderFindings(data.result?.findings || []);
      return;
    }

    activePollTimer = setTimeout(() => pollScanStatus(sessionId), 1200);
  } catch (error) {
    scanTrigger.disabled = false;
    showError(error.message);
  }
}

async function startEmailScan() {
  scanTrigger.disabled = true;
  clearError();
  clearResults();
  setProgress(0, "Starting inbox scan...");

  try {
    const response = await fetch("/api/email-scan/start", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ max_results: 0 }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Failed to start email scan.");
    }

    await pollScanStatus(data.session_id);
  } catch (error) {
    scanTrigger.disabled = false;
    showError(error.message);
  }
}

scanTrigger.addEventListener("click", startEmailScan);
window.addEventListener("load", startEmailScan);
