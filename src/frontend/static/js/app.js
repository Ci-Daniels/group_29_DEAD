document.addEventListener('DOMContentLoaded', function () {

  // --- Google Sign-In (login.html) ---
  var googleBtn = document.getElementById('google-btn');
  if (googleBtn) {
    googleBtn.addEventListener('click', function () {
      var oauthSection = document.getElementById('oauth-section');
      var authTrigger = document.getElementById('auth-trigger');
      if (oauthSection) {
        oauthSection.style.display = 'block';
        if (authTrigger) authTrigger.parentElement.style.display = 'none';
        googleBtn.style.display = 'none';
        setTimeout(function () { oauthSection.scrollIntoView({ behavior: 'smooth' }); }, 100);
      } else {
        window.location.href = '/dashboard';
      }
    });
  }

  // --- OAuth Flow (login.html) ---
  var authTrigger = document.getElementById('auth-trigger');
  var oauthSection = document.getElementById('oauth-section');
  if (authTrigger && oauthSection) {
    authTrigger.addEventListener('click', function () {
      oauthSection.style.display = 'block';
      authTrigger.style.display = 'none';
      if (googleBtn) googleBtn.style.display = 'none';
      setTimeout(function () { oauthSection.scrollIntoView({ behavior: 'smooth' }); }, 100);
    });
  }

  // --- Step Flow (verify.html) ---
  var steps = document.querySelectorAll('.step-flow');
  if (steps.length > 0) {
    var currentStep = 0;
    document.querySelectorAll('.btn-next-step').forEach(function (btn) {
      btn.addEventListener('click', function () {
        steps[currentStep].classList.remove('active');
        currentStep++;
        if (currentStep < steps.length) {
          steps[currentStep].classList.add('active');
          steps[currentStep].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      });
    });
  }

  // --- Dashboard: Confirm / Reject ---
  function updateUI() {
    var pendingItems = document.querySelectorAll('#pending-list .pending-item');
    var confirmedItems = document.querySelectorAll('#confirmed-list .asset-row');
    var pendingCount = pendingItems.length;
    var confirmedCount = confirmedItems.length;

    // Update badges
    var pendingBadge = document.getElementById('pending-badge');
    if (pendingBadge) pendingBadge.textContent = pendingCount + ' item' + (pendingCount !== 1 ? 's' : '');

    var confirmedBadge = document.getElementById('confirmed-badge');
    if (confirmedBadge) confirmedBadge.textContent = confirmedCount + ' confirmed';

    // Update stat card
    var pendingCountEl = document.getElementById('pending-count');
    if (pendingCountEl) pendingCountEl.textContent = pendingCount;

    // Update total value
    var total = 0;
    confirmedItems.forEach(function (item) {
      total += parseInt(item.getAttribute('data-value') || 0, 10);
    });
    var totalEl = document.getElementById('total-value');
    if (totalEl) totalEl.textContent = '$' + total.toLocaleString();

    // Toggle empty states
    var pendingEmpty = document.getElementById('pending-empty');
    var confirmedEmpty = document.getElementById('confirmed-empty');
    if (pendingEmpty) pendingEmpty.hidden = pendingCount > 0;
    if (confirmedEmpty) confirmedEmpty.hidden = confirmedCount > 0;
  }

  document.querySelectorAll('.btn-confirm-asset').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var pendingItem = btn.closest('.pending-item');
      var titleEl = pendingItem.querySelector('.pending-item__title');
      var subtitleEl = pendingItem.querySelector('.pending-item__subtitle');
      var assetName = titleEl ? titleEl.textContent : 'Asset';
      var assetSubtitle = subtitleEl ? subtitleEl.textContent : '';
      var assetValue = pendingItem.getAttribute('data-value') || '0';

      pendingItem.remove();

      var confirmedList = document.getElementById('confirmed-list');
      if (confirmedList) {
        var div = document.createElement('div');
        div.className = 'asset-row';
        div.setAttribute('data-value', assetValue);
        div.innerHTML =
          '<div class="asset-row__info">' +
            '<p class="asset-row__title">' + assetName + '</p>' +
            '<p class="asset-row__subtitle">Confirmed &middot; <strong>$' + Number(assetValue).toLocaleString() + '</strong></p>' +
          '</div>' +
          '<span class="badge badge--success">Confirmed</span>';
        confirmedList.prepend(div);
      }

      updateUI();
    });
  });

  document.querySelectorAll('.btn-reject-asset').forEach(function (btn) {
    btn.addEventListener('click', function () {
      btn.closest('.pending-item').remove();
      updateUI();
    });
  });

  // --- Donut Chart Hover Tooltip ---
  var tooltip = document.getElementById('donut-tooltip');
  var segments = document.querySelectorAll('.donut-segment');
  if (tooltip && segments.length > 0) {
    var wrapper = document.getElementById('donut-wrapper');
    segments.forEach(function (seg) {
      seg.addEventListener('mouseenter', function (e) {
        var label = seg.getAttribute('data-label');
        var pct = seg.getAttribute('data-pct');
        tooltip.innerHTML = '<strong>' + label + '</strong><br>' + pct + '% of assets';
        tooltip.hidden = false;
      });
      seg.addEventListener('mousemove', function (e) {
        var rect = wrapper.getBoundingClientRect();
        tooltip.style.left = (e.clientX - rect.left + 16) + 'px';
        tooltip.style.top = (e.clientY - rect.top - 16) + 'px';
      });
      seg.addEventListener('mouseleave', function () {
        tooltip.hidden = true;
      });
    });
  }

  // --- Add Next of Kin (beneficiaries.html) ---
  var addKinBtn = document.getElementById('add-kin-btn');
  if (addKinBtn) {
    addKinBtn.addEventListener('click', function () {
      var nameEl = document.getElementById('kin-name');
      var relationshipEl = document.getElementById('kin-relationship');
      var roleEl = document.getElementById('kin-role');
      var kinName = nameEl ? nameEl.value.trim() : '';
      var kinList = document.getElementById('kin-list');
      if (!kinName || !kinList) return;
      var relationship = relationshipEl ? relationshipEl.value : 'Other';
      var role = roleEl ? roleEl.value.charAt(0).toUpperCase() + roleEl.value.slice(1) : 'Viewer';
      var div = document.createElement('div');
      div.className = 'list-item';
      div.innerHTML =
        '<div>' +
          '<p class="list-item__title">' + kinName + '</p>' +
          '<p class="list-item__subtitle">' + relationship + ' &middot; ' + role + '</p>' +
        '</div>' +
        '<span class="badge badge--warning">Pending</span>';
      kinList.prepend(div);
      if (nameEl) nameEl.value = '';
    });
  }

  // --- Add Assets (beneficiaries.html) ---
  var addAssetBtn = document.getElementById('add-asset-btn');
  if (addAssetBtn) {
    addAssetBtn.addEventListener('click', function () {
      var input = document.getElementById('asset-input');
      var list = document.getElementById('asset-list');
      var empty = document.getElementById('asset-empty');
      var countBadge = document.getElementById('asset-count-badge');
      if (!input || !list) return;
      var name = input.value.trim();
      if (!name) return;

      var div = document.createElement('div');
      div.className = 'list-item';
      div.innerHTML =
        '<div>' +
          '<p class="list-item__title">' + name + '</p>' +
          '<p class="list-item__subtitle">Digital asset</p>' +
        '</div>' +
        '<button class="btn btn-ghost btn-sm asset-remove-btn" style="color:var(--danger);">Remove</button>';
      list.prepend(div);

      input.value = '';

      // Update count + toggle empty state
      updateAssetListState(list, empty, countBadge);
    });
  }

  // Remove asset (event delegation)
  document.addEventListener('click', function (e) {
    if (e.target && e.target.classList.contains('asset-remove-btn')) {
      var item = e.target.closest('.list-item');
      item.remove();
      var list = document.getElementById('asset-list');
      var empty = document.getElementById('asset-empty');
      var countBadge = document.getElementById('asset-count-badge');
      updateAssetListState(list, empty, countBadge);
    }
  });

  function updateAssetListState(list, empty, countBadge) {
    if (!list) return;
    var count = list.querySelectorAll('.list-item').length;
    if (countBadge) countBadge.textContent = count + ' asset' + (count !== 1 ? 's' : '');
    if (empty) empty.hidden = count > 0;
  }

});