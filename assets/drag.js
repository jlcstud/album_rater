(function() {
  'use strict';

  const INITIALIZED_FLAG = '__albumRatingInitialized';

  function onReady(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn);
    } else {
      fn();
    }
  }

  function extractAlbumId(gd) {
    if (!gd) return null;
    const container = gd.closest('.dash-graph');
    if (!container || !container.id) {
      return null;
    }
    try {
      const parsed = JSON.parse(container.id);
      return parsed && parsed.album ? parsed.album : null;
    } catch (err) {
      console.error('Unable to parse album id for graph', err);
      return null;
    }
  }

  function scheduleScan() {
    window.requestAnimationFrame(scanForGraphs);
  }

  function scanForGraphs() {
    const nodes = document.querySelectorAll('.js-plotly-plot');
    nodes.forEach((gd) => initialiseGraph(gd));
  }

  function initialiseGraph(gd) {
    if (!gd || gd[INITIALIZED_FLAG]) {
      return;
    }
    const albumId = extractAlbumId(gd);
    if (!albumId) {
      return;
    }
    const plotArea = gd.querySelector('.plot');
    if (!plotArea) {
      // Plotly hasn't finished rendering yet; try again shortly.
      setTimeout(() => initialiseGraph(gd), 60);
      return;
    }

    gd[INITIALIZED_FLAG] = true;
    attachHandlers(gd, albumId);
  }

  function attachHandlers(gd, albumId) {
    const state = {
      isDragging: false,
      pointerId: null,
      activeIndex: null,
      lastRating: null,
      startRating: null
    };

    function getTrace() {
      if (!gd.data || !gd.data.length) {
        return null;
      }
      return gd.data[0];
    }

    function getTrackCount() {
      const trace = getTrace();
      if (!trace || !Array.isArray(trace.x)) {
        return 0;
      }
      return trace.x.length;
    }

    function cloneCustomData() {
      const trace = getTrace();
      if (!trace || !Array.isArray(trace.customdata)) {
        return [];
      }
      return trace.customdata.map((row) => Array.isArray(row) ? row.slice() : []);
    }

    function getCurrentRating(index, rows) {
      const customRows = rows || cloneCustomData();
      if (index < 0 || index >= customRows.length) {
        return null;
      }
      const row = customRows[index];
      if (row && row.length > 2 && row[2] !== null && row[2] !== undefined) {
        const num = Number(row[2]);
        return Number.isFinite(num) ? Math.round(num * 10) / 10 : null;
      }
      return null;
    }

    function buildRatingsPayload(customRows) {
      return customRows.map((row) => {
        if (!row || row.length < 3 || row[2] === null || row[2] === undefined) {
          return null;
        }
        const num = Number(row[2]);
        return Number.isFinite(num) ? Math.round(num * 10) / 10 : null;
      });
    }

    function buildIgnoredPayload(customRows) {
      return customRows.map((row) => {
        if (!row || row.length < 2) {
          return false;
        }
        return Boolean(row[1]);
      });
    }

    function computeRating(clientY, rect) {
      if (!rect || !rect.height) {
        return 0;
      }
      const ratio = 1 - ((clientY - rect.top) / rect.height);
      let rating = 10 * ratio;
      if (!Number.isFinite(rating)) {
        rating = 0;
      }
      rating = Math.max(0, Math.min(10, rating));
      rating = Math.round(rating * 10) / 10;
      return rating;
    }

    function applyRating(index, rating) {
      if (index === null || index === undefined) {
        return;
      }
      const trace = getTrace();
      if (!trace || !Array.isArray(trace.y) || index < 0 || index >= trace.y.length) {
        return;
      }
      const newY = trace.y.slice();
      newY[index] = rating;
      const customRows = cloneCustomData();
      if (customRows[index]) {
        customRows[index][2] = rating;
      }
      state.lastRating = rating;
      trace.y = newY;
      trace.customdata = customRows;
      if (window.Plotly && typeof window.Plotly.restyle === 'function') {
        window.Plotly.restyle(gd, { y: [newY], customdata: [customRows] });
      }
    }

    function handlePointerDown(event) {
      if (event.pointerType === 'mouse' && event.button !== 0) {
        return;
      }
      const plotArea = gd.querySelector('.plot');
      if (!plotArea || !plotArea.contains(event.target)) {
        return;
      }
      const trackCount = getTrackCount();
      if (!trackCount) {
        return;
      }
      const rect = plotArea.getBoundingClientRect();
      if (!rect.width || !rect.height) {
        return;
      }
      let index = Math.floor(((event.clientX - rect.left) / rect.width) * trackCount);
      index = Math.max(0, Math.min(trackCount - 1, index));

      const customRows = cloneCustomData();
      const row = customRows[index] || [];
      const ignored = Boolean(row[1]);
      if (ignored) {
        return;
      }

      state.isDragging = true;
      state.pointerId = event.pointerId;
      state.activeIndex = index;
      state.lastRating = null;
      state.startRating = getCurrentRating(index, customRows);

      if (event.pointerType === 'mouse') {
        event.preventDefault();
      }
      plotArea.setPointerCapture && plotArea.setPointerCapture(event.pointerId);

      updateFromEvent(event, rect);

      document.addEventListener('pointermove', handlePointerMove);
      document.addEventListener('pointerup', handlePointerUp);
      document.addEventListener('pointercancel', handlePointerCancel);
    }

    function updateFromEvent(event, rect) {
      if (!state.isDragging || state.activeIndex === null) {
        return;
      }
      const plotArea = gd.querySelector('.plot');
      if (!plotArea) {
        return;
      }
      const bounds = rect || plotArea.getBoundingClientRect();
      const rating = computeRating(event.clientY, bounds);
      applyRating(state.activeIndex, rating);
    }

    function handlePointerMove(event) {
      if (!state.isDragging || (state.pointerId !== null && event.pointerId !== state.pointerId)) {
        return;
      }
      event.preventDefault();
      updateFromEvent(event);
    }

    function finishInteraction() {
      const plotArea = gd.querySelector('.plot');
      if (plotArea && state.pointerId !== null && plotArea.hasPointerCapture && plotArea.hasPointerCapture(state.pointerId)) {
        plotArea.releasePointerCapture(state.pointerId);
      }
      document.removeEventListener('pointermove', handlePointerMove);
      document.removeEventListener('pointerup', handlePointerUp);
      document.removeEventListener('pointercancel', handlePointerCancel);

      const index = state.activeIndex;
      const finalRating = state.lastRating;
      const startRating = state.startRating;

      state.isDragging = false;
      state.pointerId = null;
      state.activeIndex = null;
      state.lastRating = null;
      state.startRating = null;

      const changed = (startRating === null && finalRating !== null) ||
        (startRating !== null && finalRating === null) ||
        (startRating !== null && finalRating !== null && Math.abs(startRating - finalRating) > 1e-6);

      if (!changed || index === null || finalRating === null || finalRating === undefined) {
        return;
      }

      sendRatingUpdate();
    }

    function handlePointerUp(event) {
      if (!state.isDragging || (state.pointerId !== null && event.pointerId !== state.pointerId)) {
        return;
      }
      event.preventDefault();
      finishInteraction();
    }

    function handlePointerCancel(event) {
      if (!state.isDragging || (state.pointerId !== null && event.pointerId !== state.pointerId)) {
        return;
      }
      finishInteraction();
    }

    function sendRatingUpdate() {
      const customRows = cloneCustomData();
      const ratings = buildRatingsPayload(customRows);
      const ignored = buildIgnoredPayload(customRows);

      fetch('/api/rate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          album_id: albumId,
          ratings: ratings,
          ignored: ignored
        })
      })
        .then((res) => res.json().catch(() => null))
        .then((payload) => {
          if (!payload || !payload.ok) {
            return;
          }
          if (payload.state) {
            syncStore(payload.state);
            syncGraphFromState(payload.state);
          } else {
            syncStore({ ratings: ratings, ignored: ignored });
          }
        })
        .catch((err) => console.error('Failed to update rating', err));
    }

    function syncStore(state) {
      try {
        const storeId = JSON.stringify({ type: 'album-state', album: albumId });
        const node = document.getElementById(storeId);
        if (!node) {
          return;
        }
        const serialized = JSON.stringify(state);
        node.textContent = serialized;
        if (node.dataset) {
          node.dataset.dashStore = serialized;
        } else {
          node.setAttribute('data-dash-store', serialized);
        }
      } catch (err) {
        console.error('Failed to sync album store state', err);
      }
    }

    function syncGraphFromState(state) {
      const trace = getTrace();
      if (!trace || !Array.isArray(trace.y)) {
        return;
      }
      const ratings = Array.isArray(state.ratings) ? state.ratings : [];
      const ignored = Array.isArray(state.ignored) ? state.ignored : [];
      const newY = trace.y.slice();
      const customRows = cloneCustomData();
      for (let i = 0; i < newY.length; i += 1) {
        const ratingValue = i < ratings.length && ratings[i] !== null && ratings[i] !== undefined
          ? Number(ratings[i]) : null;
        const ignoredFlag = Boolean(i < ignored.length ? ignored[i] : false);
        newY[i] = (ratingValue === null || ignoredFlag) ? 0 : ratingValue;
        if (customRows[i]) {
          customRows[i][1] = ignoredFlag ? 1 : 0;
          customRows[i][2] = ratingValue;
        }
      }
      if (window.Plotly && typeof window.Plotly.restyle === 'function') {
        window.Plotly.restyle(gd, { y: [newY], customdata: [customRows] });
      }
    }

    const plotArea = gd.querySelector('.plot');
    if (plotArea) {
      plotArea.style.cursor = 'crosshair';
      plotArea.style.touchAction = 'none';
    }
    gd.addEventListener('pointerdown', handlePointerDown, { passive: false });
  }

  onReady(() => {
    scanForGraphs();
    const observer = new MutationObserver(scheduleScan);
    observer.observe(document.body, { childList: true, subtree: true });
  });
})();
