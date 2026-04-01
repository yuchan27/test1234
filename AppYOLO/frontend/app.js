const dom = {
  btnHealth: document.getElementById("btnHealth"),
  healthBadge: document.getElementById("healthBadge"),
  modelMeta: document.getElementById("modelMeta"),
  uploadForm: document.getElementById("uploadForm"),
  imageInput: document.getElementById("imageInput"),
  uploadBtn: document.getElementById("uploadBtn"),
  uploadStatus: document.getElementById("uploadStatus"),
  videoForm: document.getElementById("videoForm"),
  videoInput: document.getElementById("videoInput"),
  videoBtn: document.getElementById("videoBtn"),
  videoStatus: document.getElementById("videoStatus"),
  btnRunVCN: document.getElementById("btnRunVCN"),
  pipelineStatus: document.getElementById("pipelineStatus"),
  sourceInput: document.getElementById("sourceInput"),
  confInput: document.getElementById("confInput"),
  frameSkipInput: document.getElementById("frameSkipInput"),
  maxWidthInput: document.getElementById("maxWidthInput"),
  btnLiveStart: document.getElementById("btnLiveStart"),
  btnLiveStop: document.getElementById("btnLiveStop"),
  streamBadge: document.getElementById("streamBadge"),
  streamStatus: document.getElementById("streamStatus"),
  streamLogPath: document.getElementById("streamLogPath"),
  mediaLiveLayout: document.getElementById("mediaLiveLayout"),
  liveSidePanel: document.getElementById("liveSidePanel"),
  liveFrame: document.getElementById("liveFrame"),
  frameFallback: document.getElementById("frameFallback"),
  btnPreviewZoom: document.getElementById("btnPreviewZoom"),
  riskValue: document.getElementById("riskValue"),
  riskState: document.getElementById("riskState"),
  tempValue: document.getElementById("tempValue"),
  fpsValue: document.getElementById("fpsValue"),
  detCountValue: document.getElementById("detCountValue"),
  decisionAction: document.getElementById("decisionAction"),
  topConfValue: document.getElementById("topConfValue"),
  topClassValue: document.getElementById("topClassValue"),
  detectionBody: document.getElementById("detectionBody"),
  sideTempValue: document.getElementById("sideTempValue"),
  sideRiskValue: document.getElementById("sideRiskValue"),
  sideTopConfValue: document.getElementById("sideTopConfValue"),
  sideDetCountValue: document.getElementById("sideDetCountValue"),
  sideTopClassValue: document.getElementById("sideTopClassValue"),
  sideActionValue: document.getElementById("sideActionValue"),
  sideDetectionList: document.getElementById("sideDetectionList"),
  chart: document.getElementById("riskChart"),
  btnRefreshArtifacts: document.getElementById("btnRefreshArtifacts"),
  btnCleanupArtifacts: document.getElementById("btnCleanupArtifacts"),
  artifactLimit: document.getElementById("artifactLimit"),
  artifactSummary: document.getElementById("artifactSummary"),
  artifactList: document.getElementById("artifactList"),
  mediaModal: document.getElementById("mediaModal"),
  mediaModalTitle: document.getElementById("mediaModalTitle"),
  mediaModalImage: document.getElementById("mediaModalImage"),
  mediaModalDownload: document.getElementById("mediaModalDownload"),
  btnMediaClose: document.getElementById("btnMediaClose"),
};

let liveEventSource = null;
let frameTimer = null;
let riskChart = null;
let chartLabels = [];
let chartRisk = [];
let chartTemp = [];
const MAX_CHART_POINTS = 120;

function setBadge(node, text, type) {
  if (!node) {
    return;
  }
  node.textContent = text;
  node.classList.remove("neutral", "good", "alert");
  node.classList.add(type);
}

function setLiveSnapshotVisible(visible) {
  if (!dom.liveSidePanel || !dom.mediaLiveLayout) {
    return;
  }

  dom.liveSidePanel.classList.toggle("hidden", !visible);
  dom.mediaLiveLayout.classList.toggle("snapshot-hidden", !visible);
}

function bindSafe(node, eventName, handler) {
  if (!node) {
    return;
  }
  node.addEventListener(eventName, handler);
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "0";
  }
  return Number(value).toFixed(digits);
}

function formatFileSize(bytes) {
  const value = Number(bytes || 0);
  if (value <= 0) {
    return "0 B";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatTimeLabel(isoTimestamp) {
  if (!isoTimestamp) {
    return "--:--:--";
  }
  const date = new Date(isoTimestamp);
  if (Number.isNaN(date.getTime())) {
    return String(isoTimestamp);
  }
  return date.toLocaleTimeString("zh-TW", { hour12: false });
}

function createDualAxisChart(canvas, maxPoints = MAX_CHART_POINTS) {
  if (!canvas) {
    return null;
  }

  const chart = new Chart(canvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Risk Score",
          data: [],
          borderColor: "#d44e1a",
          backgroundColor: "rgba(212, 78, 26, 0.2)",
          borderWidth: 2,
          tension: 0.35,
          yAxisID: "yRisk",
          pointRadius: 0,
        },
        {
          label: "Vision Temp C",
          data: [],
          borderColor: "#006f6b",
          backgroundColor: "rgba(0, 111, 107, 0.15)",
          borderWidth: 2,
          tension: 0.3,
          yAxisID: "yTemp",
          pointRadius: 0,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          labels: {
            usePointStyle: true,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            maxTicksLimit: 8,
          },
        },
        yRisk: {
          type: "linear",
          position: "left",
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.1,
          },
        },
        yTemp: {
          type: "linear",
          position: "right",
          min: 0,
          max: 1200,
          grid: {
            drawOnChartArea: false,
          },
        },
      },
    },
  });

  chart.__maxPoints = maxPoints;
  return chart;
}

function initChart() {
  riskChart = createDualAxisChart(dom.chart, MAX_CHART_POINTS);
}

function updateCharts() {
  if (riskChart) {
    const numericTemps = chartTemp
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value));

    if (numericTemps.length > 0) {
      const tMin = Math.min(...numericTemps);
      const tMax = Math.max(...numericTemps);
      const span = Math.max(tMax - tMin, 1.0);

      const dynamicMin = Math.max(0, tMin - (span * 0.45));
      const dynamicMax = tMax + (span * 0.55);

      riskChart.options.scales.yTemp.min = Math.floor(dynamicMin / 5) * 5;
      riskChart.options.scales.yTemp.max = Math.ceil(dynamicMax / 5) * 5;
    } else {
      riskChart.options.scales.yTemp.min = 0;
      riskChart.options.scales.yTemp.max = 80;
    }

    riskChart.data.labels = chartLabels;
    riskChart.data.datasets[0].data = chartRisk;
    riskChart.data.datasets[1].data = chartTemp;
    riskChart.update("none");
  }
}

function resetChart() {
  chartLabels = [];
  chartRisk = [];
  chartTemp = [];

  updateCharts();
}

function pushChartPoint(label, risk, temp) {
  if (!riskChart) {
    return;
  }

  const maxPoints = riskChart?.__maxPoints || MAX_CHART_POINTS;
  chartLabels.push(label);
  chartRisk.push(risk);
  chartTemp.push(temp);

  while (chartLabels.length > maxPoints) {
    chartLabels.shift();
    chartRisk.shift();
    chartTemp.shift();
  }

  updateCharts();
}

function setFrameSource(src) {
  if (!dom.liveFrame || !dom.frameFallback) {
    return;
  }
  if (!src) {
    dom.liveFrame.style.display = "none";
    dom.frameFallback.style.display = "grid";
    return;
  }
  dom.liveFrame.src = src;
  dom.liveFrame.style.display = "block";
  dom.frameFallback.style.display = "none";
}

function paintDecision(decision) {
  const risk = Number(decision?.risk_score || 0);
  const alarm = Boolean(decision?.trigger_alarm);
  const action = decision?.suggested_action || "CONTINUE_MONITORING";

  if (dom.riskValue) {
    dom.riskValue.textContent = formatNumber(risk, 2);
  }
  if (dom.riskState) {
    dom.riskState.textContent = alarm ? "ALARM" : "MONITOR";
  }
  if (dom.decisionAction) {
    dom.decisionAction.textContent = action;
  }
  if (dom.sideActionValue) {
    dom.sideActionValue.textContent = action;
  }
  if (dom.sideRiskValue) {
    dom.sideRiskValue.textContent = formatNumber(risk, 2);
  }

  document.body.classList.toggle("alarm-mode", alarm);
}

function buildCurrentTelemetryRow(metrics) {
  if (!metrics) {
    return "";
  }

  const frameId = metrics.frame_id ?? "-";
  const risk = Number(metrics?.decision?.risk_score ?? metrics?.risk_score ?? 0);
  const fps = Number(metrics?.fps ?? 0);
  const timestamp = formatTimeLabel(metrics?.timestamp || "");

  const adjustedTemp = Number(metrics?.vision_temperature_celsius ?? 0);
  const rawSystemTemp = metrics?.system_temperature_celsius;
  const tempSource = String(metrics?.system_temperature_source || "unavailable");

  const tempText = rawSystemTemp === null || rawSystemTemp === undefined
    ? `${formatNumber(adjustedTemp, 2)} °C (${tempSource})`
    : `${formatNumber(rawSystemTemp, 2)} -> ${formatNumber(adjustedTemp, 2)} °C (${tempSource})`;

  const detail = `frame=${frameId}, risk=${formatNumber(risk, 2)}, fps=${formatNumber(fps, 1)}, time=${timestamp}`;
  return `<tr><td>CURRENT</td><td>${escapeHtml(tempText)}</td><td>${escapeHtml(detail)}</td></tr>`;
}

function renderDetectionDetails(detections, metrics = null) {
  const rows = Array.isArray(detections) ? detections : [];
  const currentRowHtml = buildCurrentTelemetryRow(metrics);

  if (dom.topConfValue) {
    dom.topConfValue.textContent = "0%";
  }
  if (dom.topClassValue) {
    dom.topClassValue.textContent = "No detection (current frame)";
  }
  if (dom.sideTopConfValue) {
    dom.sideTopConfValue.textContent = "0%";
  }
  if (dom.sideTopClassValue) {
    dom.sideTopClassValue.textContent = "No detection (current frame)";
  }

  if (!dom.detectionBody) {
    // Continue to keep side panel functional even if table body is absent.
  }

  if (rows.length === 0) {
    if (dom.detectionBody) {
      const emptyRow = "<tr><td colspan='3' class='empty-cell'>No detections in current frame</td></tr>";
      dom.detectionBody.innerHTML = `${currentRowHtml}${emptyRow}`;
    }
    if (dom.sideDetectionList) {
      dom.sideDetectionList.innerHTML = "<li class='side-empty'>No detections in current frame</li>";
    }
    return;
  }

  const sorted = [...rows].sort((a, b) => Number(b?.confidence || 0) - Number(a?.confidence || 0));
  const top = sorted[0];
  if (dom.topConfValue) {
    dom.topConfValue.textContent = `${formatNumber(Number(top?.confidence || 0) * 100, 1)}%`;
  }
  if (dom.topClassValue) {
    dom.topClassValue.textContent = String(top?.class_name || "unknown");
  }
  if (dom.sideTopConfValue) {
    dom.sideTopConfValue.textContent = `${formatNumber(Number(top?.confidence || 0) * 100, 1)}%`;
  }
  if (dom.sideTopClassValue) {
    dom.sideTopClassValue.textContent = String(top?.class_name || "unknown");
  }

  const limited = sorted.slice(0, 8);
  const html = limited
    .map((item) => {
      const cls = escapeHtml(item?.class_name || "unknown");
      const conf = `${formatNumber(Number(item?.confidence || 0) * 100, 1)}%`;
      const bbox = Array.isArray(item?.bbox) ? item.bbox.map((v) => formatNumber(v, 2)).join(", ") : "-";
      return `<tr><td>${cls}</td><td>${conf}</td><td>${bbox}</td></tr>`;
    })
    .join("");

  if (dom.detectionBody) {
    dom.detectionBody.innerHTML = `${currentRowHtml}${html}`;
  }

  if (dom.sideDetectionList) {
    const sideHtml = limited
      .slice(0, 5)
      .map((item) => {
        const cls = escapeHtml(item?.class_name || "unknown");
        const conf = `${formatNumber(Number(item?.confidence || 0) * 100, 1)}%`;
        return `<li><span class='side-det-class'>${cls}</span><span class='side-det-conf'>${conf}</span></li>`;
      })
      .join("");
    dom.sideDetectionList.innerHTML = sideHtml;
  }
}

function paintMetrics(metrics) {
  if (!metrics) {
    return;
  }

  const detectionRows = Array.isArray(metrics.detections) ? metrics.detections : [];
  const currentDetectionCount = detectionRows.length;

  if (dom.fpsValue) {
    dom.fpsValue.textContent = formatNumber(metrics.fps, 1);
  }
  if (dom.detCountValue) {
    dom.detCountValue.textContent = String(currentDetectionCount);
  }
  if (dom.sideDetCountValue) {
    dom.sideDetCountValue.textContent = String(currentDetectionCount);
  }

  renderDetectionDetails(detectionRows, metrics);

  const temp = metrics.vision_temperature_celsius;
  if (dom.tempValue) {
    dom.tempValue.textContent = temp === null || temp === undefined ? "0" : formatNumber(temp, 2);
  }
  if (dom.sideTempValue) {
    const sideTemp = temp === null || temp === undefined ? 0 : Number(temp);
    dom.sideTempValue.textContent = `${formatNumber(sideTemp, 2)} °C`;
  }

  paintDecision(metrics.decision || {});

  const risk = Number(metrics.decision?.risk_score || 0);
  const chartTempValue = temp === null || temp === undefined ? 0 : Number(temp);
  const label = formatTimeLabel(metrics.timestamp);
  pushChartPoint(label, risk, chartTempValue);
}

function toPointTimeLabel(point, index = 0) {
  if (point?.timestamp) {
    return formatTimeLabel(point.timestamp);
  }

  if (point?.time_sec !== null && point?.time_sec !== undefined) {
    const sec = Number(point.time_sec);
    if (Number.isFinite(sec)) {
      return `T+${sec.toFixed(1)}s`;
    }
  }

  const frameId = Number(point?.frame_id);
  if (Number.isFinite(frameId)) {
    return `T+${(frameId / 30).toFixed(1)}s`;
  }

  return `T+${(index / 5).toFixed(1)}s`;
}

function paintVideoTelemetry(video) {
  if (!video) {
    return;
  }

  const points = Array.isArray(video.telemetry?.points) ? video.telemetry.points : [];
  if (points.length > 0) {
    resetChart();
    for (const [index, point] of points.entries()) {
      const frameLabel = toPointTimeLabel(point, index);
      const risk = Number(point.risk_score || 0);
      const temp = Number(point.vision_temperature_celsius || 0);
      pushChartPoint(frameLabel, risk, temp);
    }

    const last = points[points.length - 1];
    paintDecision({
      risk_score: Number(last.risk_score || 0),
      trigger_alarm: Boolean(last.trigger_alarm),
      suggested_action: last.suggested_action || "CONTINUE_MONITORING",
    });

    if (dom.detCountValue) {
      dom.detCountValue.textContent = String(last.detection_count || 0);
    }
    if (dom.tempValue) {
      dom.tempValue.textContent = formatNumber(last.vision_temperature_celsius || 0, 2);
    }
  }

  if (dom.fpsValue) {
    dom.fpsValue.textContent = "0";
  }

  // Video telemetry points do not contain per-box confidence details.
  const lastPoint = points.length > 0 ? points[points.length - 1] : null;
  renderDetectionDetails([], {
    frame_id: lastPoint?.frame_id,
    timestamp: lastPoint?.timestamp,
    vision_temperature_celsius: lastPoint?.vision_temperature_celsius,
    system_temperature_celsius: lastPoint?.system_temperature_celsius,
    system_temperature_source: lastPoint?.system_temperature_source || "video-telemetry",
    fps: 0,
    risk_score: lastPoint?.risk_score,
    decision: { risk_score: Number(lastPoint?.risk_score || 0) },
  });

  const preview = video.preview_image_base64;
  if (preview) {
    setFrameSource(`data:image/jpeg;base64,${preview}`);
  }
}

function renderLiveHistoryFromState(state) {
  const history = state?.history || {};
  const risks = Array.isArray(history.risk) ? history.risk : [];
  const temps = Array.isArray(history.temperature) ? history.temperature : [];
  const timestamps = Array.isArray(history.timestamps) ? history.timestamps : [];

  if (risks.length === 0 && temps.length === 0) {
    return;
  }

  resetChart();
  const length = Math.max(risks.length, temps.length, timestamps.length);
  for (let i = 0; i < length; i += 1) {
    const label = timestamps[i] ? formatTimeLabel(timestamps[i]) : `T+${(i / 5).toFixed(1)}s`;
    const risk = Number(risks[i] || 0);
    const temp = Number(temps[i] || 0);
    pushChartPoint(label, risk, temp);
  }
}

function openMediaModal({ src, title, downloadHref = "", isMjpeg = false }) {
  if (!dom.mediaModal || !dom.mediaModalImage) {
    return;
  }
  if (!src) {
    return;
  }

  dom.mediaModal.classList.remove("hidden");
  dom.mediaModal.setAttribute("aria-hidden", "false");

  if (dom.mediaModalTitle) {
    dom.mediaModalTitle.textContent = title || "Preview";
  }

  if (dom.mediaModalDownload) {
    if (downloadHref) {
      dom.mediaModalDownload.href = downloadHref;
      dom.mediaModalDownload.classList.remove("hidden");
    } else {
      dom.mediaModalDownload.href = "#";
      dom.mediaModalDownload.classList.add("hidden");
    }
  }

  const sourceWithBust = isMjpeg ? `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}` : src;
  dom.mediaModalImage.src = sourceWithBust;
}

function closeMediaModal() {
  if (!dom.mediaModal || !dom.mediaModalImage) {
    return;
  }

  dom.mediaModal.classList.add("hidden");
  dom.mediaModal.setAttribute("aria-hidden", "true");
  dom.mediaModalImage.src = "";
}

function renderArtifacts(files) {
  if (!dom.artifactList) {
    return;
  }

  const visualFiles = (Array.isArray(files) ? files : []).filter((file) => file?.kind === "image" || file?.kind === "video");

  if (visualFiles.length === 0) {
    dom.artifactList.innerHTML = "<p class='helper'>No previewable image/video files found.</p>";
    if (dom.artifactSummary) {
      dom.artifactSummary.textContent = "No previewable image/video files found.";
    }
    return;
  }

  const rows = visualFiles.map((file) => {
    const encodedPath = encodeURIComponent(file.relative_path || "");
    const safeUrl = escapeHtml(file.url || "");
    const safeName = escapeHtml(file.name || "artifact");

    let preview = "<div class='artifact-fallback'>No preview</div>";
    if (file.kind === "image") {
      preview = `<img src='${file.url}' alt='${file.name}' loading='lazy' />`;
    } else if (file.kind === "video") {
      preview = `<img src='/api/video/thumbnail?path=${encodedPath}' alt='${safeName}' loading='lazy' />`;
    }

    const openPreviewButton = `<button
      class='artifact-link artifact-open-btn'
      type='button'
      data-kind='${file.kind}'
      data-name='${safeName}'
      data-url='${safeUrl}'
      data-path='${encodedPath}'
    >Open Preview</button>`;
    const downloadLink = `<a class='artifact-link' href='${safeUrl}' target='_blank' rel='noopener noreferrer'>Download</a>`;

    return `
      <article class='artifact-item'>
        <div class='artifact-preview'>${preview}</div>
        <div class='artifact-meta'>
          <div class='artifact-name' title='${safeName}'>${safeName}</div>
          <div class='artifact-detail'>${file.kind.toUpperCase()} | ${formatFileSize(file.size_bytes)}</div>
          ${openPreviewButton}
          ${downloadLink}
        </div>
      </article>
    `;
  });

  dom.artifactList.innerHTML = rows.join("");
  if (dom.artifactSummary) {
    dom.artifactSummary.textContent = `Showing ${visualFiles.length} previewable files.`;
  }
}

async function refreshArtifacts() {
  const limit = Number(dom.artifactLimit?.value || 48);

  try {
    const response = await fetch(`/api/generated/files?limit=${limit}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    renderArtifacts(payload.files || []);
  } catch (error) {
    if (dom.artifactList) {
      dom.artifactList.innerHTML = "<p class='helper'>Failed to load generated artifacts.</p>";
    }
    if (dom.artifactSummary) {
      dom.artifactSummary.textContent = "Failed to load generated files.";
    }
  }
}

async function cleanupArtifacts() {
  if (dom.artifactSummary) {
    dom.artifactSummary.textContent = "Cleaning old generated files...";
  }

  const keepLatest = Number(dom.artifactLimit?.value || 48);
  try {
    const response = await fetch(`/api/generated/files/cleanup?keep_latest=${keepLatest}`, {
      method: "POST",
    });
    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    const payload = await response.json();
    renderArtifacts(payload.files || []);
    if (dom.artifactSummary) {
      dom.artifactSummary.textContent = `Cleanup done. Removed ${payload.deleted_count || 0} files.`;
    }
  } catch (error) {
    if (dom.artifactSummary) {
      dom.artifactSummary.textContent = `Cleanup failed: ${error.message}`;
    }
  }
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    if (payload.model_ready) {
      setBadge(dom.healthBadge, "READY", "good");
    } else {
      setBadge(dom.healthBadge, "DEGRADED", "alert");
    }
  } catch (error) {
    setBadge(dom.healthBadge, "OFFLINE", "alert");
  }
}

async function refreshModelMeta() {
  try {
    const response = await fetch("/api/model/info");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const payload = await response.json();
    const classes = payload.classes || {};
    const classCount = Object.keys(classes).length;
    const modelPath = payload.model_path || "unknown";
    if (dom.modelMeta) {
      dom.modelMeta.textContent = `Model: ${modelPath} | Classes: ${classCount}`;
    }
  } catch (error) {
    if (dom.modelMeta) {
      dom.modelMeta.textContent = "Model metadata unavailable";
    }
  }
}

function stopFrameLoop() {
  if (frameTimer) {
    clearInterval(frameTimer);
    frameTimer = null;
  }
}

function startFrameLoop() {
  stopFrameLoop();
  frameTimer = setInterval(() => {
    if (dom.liveFrame) {
      dom.liveFrame.src = `/api/live/frame?t=${Date.now()}`;
    }
  }, 240);
}

function disconnectEvents() {
  if (liveEventSource) {
    liveEventSource.close();
    liveEventSource = null;
  }
}

function connectEvents() {
  disconnectEvents();
  liveEventSource = new EventSource("/api/live/events");

  liveEventSource.onmessage = (event) => {
    let state = null;
    try {
      state = JSON.parse(event.data);
    } catch (error) {
      return;
    }

    if (!state) {
      return;
    }

    if (dom.streamLogPath) {
      const logPath = state.current_log_path || "not started";
      dom.streamLogPath.textContent = `Live log: ${logPath}`;
    }

    if (state.last_error) {
      dom.streamStatus.textContent = state.last_error;
      setBadge(dom.streamBadge, "WARN", "alert");
    }

    if (state.running) {
      setBadge(dom.streamBadge, "LIVE", "good");
      dom.streamStatus.textContent = "Receiving live telemetry.";
    }

    setLiveSnapshotVisible(Boolean(state.running));

    if (state.latest_metrics && Object.keys(state.latest_metrics).length > 0) {
      paintMetrics(state.latest_metrics);
    }
  };

  liveEventSource.onerror = () => {
    setBadge(dom.streamBadge, "RETRY", "alert");
    dom.streamStatus.textContent = "SSE disconnected, reconnecting...";
  };
}

async function startLive() {
  if (!dom.sourceInput || !dom.confInput || !dom.frameSkipInput || !dom.maxWidthInput) {
    return;
  }
  const source = (dom.sourceInput.value || "0").trim() || "0";
  const conf = Number(dom.confInput.value || 0.25);
  const frameSkip = Math.max(1, Math.min(8, Number(dom.frameSkipInput.value || 1)));
  const maxFrameWidth = Math.max(640, Math.min(3840, Number(dom.maxWidthInput.value || 1280)));

  try {
    const response = await fetch("/api/live/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source,
        conf,
        frame_skip: frameSkip,
        max_frame_width: maxFrameWidth,
      }),
    });

    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    setBadge(dom.streamBadge, "LIVE", "good");
    dom.streamStatus.textContent = `Live stream started: source=${source}, skip=${frameSkip}, maxWidth=${maxFrameWidth}`;

    setLiveSnapshotVisible(true);
    resetChart();
    connectEvents();
    startFrameLoop();
  } catch (error) {
    setBadge(dom.streamBadge, "ERROR", "alert");
    dom.streamStatus.textContent = `Start failed: ${error.message}`;
  }
}

async function stopLive() {
  try {
    await fetch("/api/live/stop", { method: "POST" });
  } catch (error) {
    // Keep UI fallback even if API fails.
  }

  disconnectEvents();
  stopFrameLoop();
  setLiveSnapshotVisible(false);
  setBadge(dom.streamBadge, "IDLE", "neutral");
  dom.streamStatus.textContent = "Live stream stopped.";
}

async function runImageInference(event) {
  event.preventDefault();

  if (!dom.imageInput || !dom.uploadBtn || !dom.uploadStatus) {
    return;
  }

  setLiveSnapshotVisible(false);

  const file = dom.imageInput.files[0];
  if (!file) {
    dom.uploadStatus.textContent = "Please choose an image file first.";
    return;
  }

  dom.uploadBtn.disabled = true;
  dom.uploadStatus.textContent = "Running inference...";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("save_annotated", "true");

  try {
    const response = await fetch("/api/inference/image", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    const payload = await response.json();
    const inference = payload.inference;

    paintMetrics({
      fps: 0,
      detection_count: inference.detection_count,
      vision_temperature_celsius: inference.vision_temperature_celsius,
      decision: inference.decision,
      detections: inference.detections,
      frame_id: inference.frame_id,
    });

    if (inference.annotated_image_base64) {
      setFrameSource(`data:image/jpeg;base64,${inference.annotated_image_base64}`);
    }

    const action = inference.decision?.suggested_action || "CONTINUE_MONITORING";
    dom.uploadStatus.textContent = `Inference done. Action: ${action}`;
    await refreshArtifacts();
  } catch (error) {
    dom.uploadStatus.textContent = `Inference failed: ${error.message}`;
  } finally {
    dom.uploadBtn.disabled = false;
  }
}

async function runVideoInference(event) {
  event.preventDefault();

  if (!dom.videoInput || !dom.videoBtn || !dom.videoStatus) {
    return;
  }

  setLiveSnapshotVisible(false);

  const file = dom.videoInput.files[0];
  if (!file) {
    dom.videoStatus.textContent = "Please choose a video file first.";
    return;
  }

  dom.videoBtn.disabled = true;
  dom.videoStatus.textContent = "Running video inference... this may take a while.";

  const formData = new FormData();
  formData.append("file", file);
  formData.append("with_decision", "true");

  try {
    const response = await fetch("/api/inference/video", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    const payload = await response.json();
    const video = payload.video || {};
    paintVideoTelemetry(video);

    const playerUrl = video.player_url || "";
    dom.videoStatus.textContent = `Video done. Frames: ${video.frame_count || 0}, Alarm frames: ${video.alarm_frame_count || 0}${playerUrl ? ` | Player: ${playerUrl}` : ""}`;
    await refreshArtifacts();
  } catch (error) {
    dom.videoStatus.textContent = `Video inference failed: ${error.message}`;
  } finally {
    dom.videoBtn.disabled = false;
  }
}

async function runVCNPipeline() {
  if (!dom.pipelineStatus || !dom.btnRunVCN) {
    return;
  }

  setLiveSnapshotVisible(false);

  dom.pipelineStatus.textContent = "Running VCN.py logic...";
  dom.btnRunVCN.disabled = true;

  try {
    const response = await fetch("/api/pipeline/vcn/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });

    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    const payload = await response.json();
    const count = (payload.pipeline?.processed_files || []).length;
    const outputUrl = payload.pipeline?.output_url;
    if (outputUrl) {
      setFrameSource(`${outputUrl}?t=${Date.now()}`);
    }
    dom.pipelineStatus.textContent = `VCN logic done. Processed cameras: ${count}`;
    await refreshArtifacts();
  } catch (error) {
    dom.pipelineStatus.textContent = `VCN pipeline failed: ${error.message}`;
  } finally {
    dom.btnRunVCN.disabled = false;
  }
}

async function restoreLiveState() {
  try {
    const response = await fetch("/api/live/state");
    if (!response.ok) {
      return;
    }

    const state = await response.json();
    renderLiveHistoryFromState(state);
    setLiveSnapshotVisible(Boolean(state.running));

    if (state.running) {
      setBadge(dom.streamBadge, "LIVE", "good");
      dom.streamStatus.textContent = "Recovered existing live stream state.";
      connectEvents();
      startFrameLoop();
    }

    if (dom.streamLogPath) {
      const logPath = state.current_log_path || "not started";
      dom.streamLogPath.textContent = `Live log: ${logPath}`;
    }

    if (state.latest_metrics && Object.keys(state.latest_metrics).length > 0) {
      paintMetrics(state.latest_metrics);
    }
  } catch (error) {
    // Keep quiet during startup.
  }
}

function openLivePreviewZoom() {
  if (!dom.liveFrame || !dom.liveFrame.src) {
    return;
  }

  openMediaModal({
    src: dom.liveFrame.src,
    title: "Annotated Preview",
    downloadHref: dom.liveFrame.src,
  });
}

function onArtifactListClick(event) {
  const button = event.target.closest(".artifact-open-btn");
  if (!button) {
    return;
  }

  const kind = button.dataset.kind || "other";
  const name = button.dataset.name || "Artifact";
  const fileUrl = button.dataset.url || "";
  const encodedPath = button.dataset.path || "";
  const relativePath = decodeURIComponent(encodedPath || "");

  if (kind === "video" && relativePath) {
    openMediaModal({
      src: `/api/video/mjpeg?path=${encodeURIComponent(relativePath)}&fps=15&loop=true`,
      title: `${name} (Embedded Player)`,
      downloadHref: fileUrl,
      isMjpeg: true,
    });
    return;
  }

  if (fileUrl) {
    openMediaModal({
      src: fileUrl,
      title: name,
      downloadHref: fileUrl,
    });
  }
}

function bindEvents() {
  bindSafe(dom.btnHealth, "click", refreshHealth);
  bindSafe(dom.uploadForm, "submit", runImageInference);
  bindSafe(dom.videoForm, "submit", runVideoInference);
  bindSafe(dom.btnRunVCN, "click", runVCNPipeline);
  bindSafe(dom.btnRefreshArtifacts, "click", refreshArtifacts);
  bindSafe(dom.btnCleanupArtifacts, "click", cleanupArtifacts);
  bindSafe(dom.artifactLimit, "change", refreshArtifacts);
  bindSafe(dom.btnLiveStart, "click", startLive);
  bindSafe(dom.btnLiveStop, "click", stopLive);
  bindSafe(dom.btnPreviewZoom, "click", openLivePreviewZoom);
  bindSafe(dom.artifactList, "click", onArtifactListClick);
  bindSafe(dom.btnMediaClose, "click", closeMediaModal);
  bindSafe(dom.mediaModal, "click", (event) => {
    if (event.target === dom.mediaModal) {
      closeMediaModal();
    }
  });

  bindSafe(dom.liveFrame, "error", () => {
    setFrameSource("");
  });

  bindSafe(dom.liveFrame, "load", () => {
    if (dom.liveFrame && dom.liveFrame.src) {
      dom.liveFrame.style.display = "block";
      if (dom.frameFallback) {
        dom.frameFallback.style.display = "none";
      }
    }
  });
  bindSafe(dom.liveFrame, "click", openLivePreviewZoom);
}

function init() {
  try {
    setLiveSnapshotVisible(false);
    initChart();
    bindEvents();
    refreshHealth();
    refreshModelMeta();
    restoreLiveState();
    refreshArtifacts();
    setInterval(refreshHealth, 8000);
  } catch (error) {
    if (dom.pipelineStatus) {
      dom.pipelineStatus.textContent = `UI init error: ${error.message}`;
    }
  }
}

init();

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeMediaModal();
  }
});

window.addEventListener("error", (event) => {
  if (dom.pipelineStatus) {
    dom.pipelineStatus.textContent = `Frontend error: ${event.message}`;
  }
});
