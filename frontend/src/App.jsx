import { useState, useEffect, useCallback, useRef } from "react";
import {
  MapContainer, TileLayer, ImageOverlay,
  LayersControl, ScaleControl, GeoJSON, useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// The backend sleeps when idle on the free tier and can take 20 to 30 seconds to
// wake. Every call therefore carries an explicit timeout, and the calls made at
// startup are retried, so a cold server no longer leaves the map permanently
// empty or the spinner running forever.
async function apiFetch(path, opts = {}) {
  const { timeout = 20000, retries = 0, retryDelay = 1500, ...init } = opts;
  let lastErr;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const res = await fetch(`${API_BASE_URL}${path}`, { ...init, signal: controller.signal });
      clearTimeout(timer);
      if (!res.ok) {
        let detail = "";
        try { detail = (await res.json())?.detail || ""; } catch { /* non-JSON body */ }
        throw new Error(detail || `Server returned ${res.status} ${res.statusText}`);
      }
      return res;
    } catch (err) {
      clearTimeout(timer);
      // A fetch rejection here is usually the backend still starting up rather
      // than a genuine cross-origin misconfiguration; say something the user can
      // act on instead of surfacing the browser's raw wording.
      lastErr = err?.name === "AbortError"
        ? new Error(`Request timed out after ${Math.round(timeout / 1000)}s`)
        : (err instanceof TypeError
            ? new Error("Could not reach the prediction server. It may still be starting up; try again in a moment.")
            : err);
      if (attempt < retries) {
        await new Promise((r) => setTimeout(r, retryDelay * (attempt + 1)));
      }
    }
  }
  throw lastErr;
}

// ── Local scenario history ───────────────────────────────────────────────────
// The prediction cache on the server is deliberately shared: its whole value is
// that one person's run makes the same run instant for everyone else. History is
// the opposite — it should be the scenarios YOU ran, so this is kept per browser.
//
// Only the three inputs and a timestamp are stored, never the outputs. Output
// files are rotated server-side, so a stored result would eventually point at
// files that no longer exist. Loading an entry re-requests it instead, which the
// shared cache usually answers immediately.
const HISTORY_KEY = "mlbaha.history.v1";
const HISTORY_MAX = 20;

function readHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    const list = raw ? JSON.parse(raw) : [];
    return Array.isArray(list) ? list : [];
  } catch {
    return [];   // private browsing, blocked storage, or corrupt value
  }
}

function writeHistory(list) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, HISTORY_MAX)));
  } catch {
    // Storage unavailable or full. History is a convenience; carry on without it.
  }
}

const HAZARD_NAMES = {
  0: "No Hazard", 1: "Low",      2: "Moderate",
  3: "High",      4: "Very High", 5: "Extreme",
};

const DEPTH_RANGES = {
  0: "No inundation", 1: "0.0 – 0.5 m", 2: "0.5 – 1.0 m",
  3: "1.0 – 1.5 m",   4: "1.5 – 2.0 m", 5: "> 2.0 m",
};

const HAZARD_COLORS = {
  0: "#374151", 1: "#facc15", 2: "#f97316",
  3: "#ea580c", 4: "#dc2626", 5: "#991b1b",
};

// Text colour to place ON each hazard colour. White was used for every class,
// which gives 1.53:1 on Low and 2.80:1 on Moderate, both well under the 4.5:1
// WCAG AA threshold. Measured ratios with these values: Low 11.96, Moderate
// 6.53, High 5.14, Very High 4.83, Extreme 8.31, No Hazard 10.31.
const HAZARD_ON_COLOR = {
  0: "#ffffff", 1: "#0a1525", 2: "#0a1525",
  3: "#0a1525", 4: "#ffffff", 5: "#ffffff",
};

const HOTLINES = [
  { label: "MDRRMO Sipocot", number: "0907-030-5000", icon: "🚨" },
  { label: "BFP Sipocot",    number: "0999-938-0063", icon: "🚒" },
  { label: "PNP Sipocot",    number: "0998-598-5975", icon: "👮" },
  { label: "MHO Sipocot",    number: "0998-979-5783", icon: "🏥" },
  { label: "MSWDO Sipocot",  number: "0917-854-5409", icon: "🏛" },
];

// ── UTM Zone 51N → WGS84 (Bowring approximation) ─────────────────────────────
function utmToWgs84Ref(easting, northing) {
  const k0 = 0.9996, a = 6378137, e2 = 0.00669438;
  const e1  = (1 - Math.sqrt(1-e2)) / (1 + Math.sqrt(1-e2));
  const x   = easting - 500000;
  const y   = northing;
  const M   = y / k0;
  const mu  = M / (a * (1 - e2/4 - 3*e2*e2/64));
  const p1  = mu + (3*e1/2 - 27*e1*e1*e1/32) * Math.sin(2*mu);
  const p2  = p1 + (21*e1*e1/16 - 55*e1*e1*e1*e1/32) * Math.sin(4*mu);
  const p3  = p2 + (151*e1*e1*e1/96) * Math.sin(6*mu);
  const lat1= p3;
  const N1  = a / Math.sqrt(1 - e2*Math.sin(lat1)**2);
  const T1  = Math.tan(lat1)**2;
  const C1  = e2*Math.cos(lat1)**2 / (1-e2);
  const R1  = a*(1-e2) / Math.pow(1-e2*Math.sin(lat1)**2, 1.5);
  const D   = x / (N1*k0);
  const lat = lat1 - (N1*Math.tan(lat1)/R1)*(D*D/2-(5+3*T1+10*C1-4*C1*C1-9*e2)*D*D*D*D/24);
  const lon0= ((51-1)*6-180+3)*Math.PI/180;
  const lon = lon0 + (D-(1+2*T1+C1)*D*D*D/6)/Math.cos(lat1);
  return [lon*180/Math.PI, lat*180/Math.PI];
}
function MapFitter({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [24, 24] });
  }, [bounds, map]);
  return null;
}

function BarangayFocuser({ feature }) {
  const map = useMap();
  useEffect(() => {
    if (!feature) return;
    const L = window.L;
    if (!L) return;
    const layer = L.geoJSON(feature);
    map.fitBounds(layer.getBounds(), { padding: [40, 40] });
  }, [feature, map]);
  return null;
}

// Dismiss barangay popup when clicking empty map area
function MapClickDismiss({ onDismiss }) {
  const map = useMap();
  useEffect(() => {
    map.on("click", onDismiss);
    return () => map.off("click", onDismiss);
  }, [map, onDismiss]);
  return null;
}
function MapHoverTooltip({ overlayBounds, csvCells }) {
  const map = useMap();
  const [tooltip, setTooltip] = useState(null);

  const active = Boolean(overlayBounds && csvCells?.length);

  useEffect(() => {
    if (!active) return undefined;

    const onMove = (e) => {
      const { lat, lng } = e.latlng;
      const [[s, w], [n, ee]] = overlayBounds;
      if (lat < s || lat > n || lng < w || lng > ee) {
        setTooltip(null); return;
      }
      // Find nearest cell
      let best = null, bestDist = Infinity;
      for (const c of csvCells) {
        const d = (c.lat - lat) ** 2 + (c.lng - lng) ** 2;
        if (d < bestDist) { bestDist = d; best = c; }
      }
      if (best) {
        const pt = map.latLngToContainerPoint([lat, lng]);
        setTooltip({ x: pt.x, y: pt.y, code: best.code, depth: best.depth });
      }
    };

    const onOut = () => setTooltip(null);
    map.on("mousemove", onMove);
    map.on("mouseout",  onOut);
    return () => {
      map.off("mousemove", onMove);
      map.off("mouseout",  onOut);
      setTooltip(null);
    };
  }, [map, active, overlayBounds, csvCells]);

  if (!active || !tooltip) return null;
  const name  = HAZARD_NAMES[tooltip.code] ?? "Unknown";
  const color = HAZARD_COLORS[tooltip.code] ?? "#888";
  const depth = tooltip.depth > 0 ? ` · ${tooltip.depth.toFixed(2)} m` : "";
  return (
    <div className="map-hover-tooltip" style={{ left: tooltip.x + 14, top: tooltip.y - 10 }}>
      <span className="map-hover-dot" style={{ background: color }} />
      <span>{name}{depth}</span>
    </div>
  );
}
function HazardSwatch({ color }) {
  return <span className="legend-swatch" style={{ background: color || "#ccc" }} />;
}

function DownloadBtn({ href, label, icon }) {
  if (!href) return null;
  return (
    <a className="dl-btn" href={href} download target="_blank" rel="noreferrer">
      <span className="dl-icon">{icon}</span>{label}
    </a>
  );
}

function ScenarioRow({ s, onLoad }) {
  return (
    <div className="scenario-row">
      <div className="scenario-meta">
        <span className="scenario-ts">{new Date(s.ts).toLocaleString()}</span>
        <span className="scenario-params">{s.depth}mm / {s.duration}h / API:{s.antecedent}mm</span>
      </div>
      <button className="scenario-load-btn" onClick={() => onLoad(s)}
        aria-label={`Re-run ${s.depth} millimetres over ${s.duration} hours`}>Load</button>
    </div>
  );
}

function LoadingOverlay({ waking }) {
  return (
    <div className="loading-overlay" aria-live="polite">
      <div className="spinner-ring" />
      <p>{waking ? "Waking the prediction server…" : "Running ML-BaHa prediction…"}</p>
      <p className="loading-sub">
        {waking
          ? "The server sleeps when idle, so the first run after a pause takes 20 to 30 seconds."
          : "Computing flood depth for all grid cells"}
      </p>
    </div>
  );
}

const TAB_LABELS = {
  inputs:    "⚙ Inputs",
  legend:    "🎨 Legend",
  summary:   "📊 Summary",
  history:   "🕘 History",
  downloads: "⬇ Export",
  about:     "ℹ About",
};

// ── GeoJSON styles ────────────────────────────────────────────────────────────
const barangayStyle = {
  color: "#67e8f9", weight: 1.5,
  fillColor: "transparent", fillOpacity: 0, dashArray: "",
};
const barangayHighlightStyle = {
  color: "#ffffff", weight: 3,
  fillColor: "transparent", fillOpacity: 0, dashArray: "",
};

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [duration,   setDuration]   = useState(12);
  const [depth,      setDepth]      = useState(180);
  const [antecedent, setAntecedent] = useState(60);

  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");
  const [scenarios, setScenarios] = useState(() => readHistory());

  const [opacity,          setOpacity]          = useState(0.75);
  const [overlayVisible,   setOverlayVisible]   = useState(true);
  const [barangayVisible,  setBarangayVisible]  = useState(true);
  const [activeTab,        setActiveTab]        = useState("inputs");
  const [sheetOpen,        setSheetOpen]        = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const [barangayGeoJSON,  setBarangayGeoJSON]  = useState(null);
  const [barangayList,     setBarangayList]     = useState([]);
  const [barangaySearch,   setBarangaySearch]   = useState("");
  const [selectedBarangay, setSelectedBarangay] = useState(null);
  const [highlightedName,  setHighlightedName]  = useState("");
  const [summarySearch,    setSummarySearch]    = useState("");
  const [barangayPopup,    setBarangayPopup]    = useState(null); // { name, x, y }

  const geojsonRef  = useRef(null);
  const [csvCells, setCsvCells] = useState([]); // lat/lng cells for hover tooltip
  const [waking,    setWaking]    = useState(false); // backend cold start in progress
  const [timing,    setTiming]    = useState(null);  // { ms, cached } of the last run
  const [bootError, setBootError] = useState("");

  const mapOutputs = result?.map_outputs;
  const overlayUrl = mapOutputs?.hazard_png ? `${API_BASE_URL}${mapOutputs.hazard_png}` : null;
  const bounds     = mapOutputs?.bounds || null;

  const outputUrls = {
    depthRaster:   result?.outputs?.depth_raster   ? `${API_BASE_URL}${result.outputs.depth_raster}`   : null,
    hazardRaster:  result?.outputs?.hazard_raster  ? `${API_BASE_URL}${result.outputs.hazard_raster}`  : null,
    predictionCsv: result?.outputs?.prediction_csv ? `${API_BASE_URL}${result.outputs.prediction_csv}` : null,
    summaryJson:   result?.outputs?.summary_json   ? `${API_BASE_URL}${result.outputs.summary_json}`   : null,
  };

  // ── Startup: wake the backend, then load barangay boundaries ─────────────
  useEffect(() => {
    let cancelled = false;
    // Only announce the cold start if it actually is one; a warm server answers
    // well inside this window and the user never sees the notice.
    const slowTimer = setTimeout(() => { if (!cancelled) setWaking(true); }, 5000);

    (async () => {
      try {
        await apiFetch("/api/health", { timeout: 70000, retries: 2, retryDelay: 3000 });
      } catch {
        if (!cancelled) {
          setBootError(
            "Could not reach the prediction server. It may still be starting up, so try Run Prediction in a moment."
          );
        }
      }
      clearTimeout(slowTimer);
      if (!cancelled) setWaking(false);
      if (cancelled) return;

      try {
        const res = await apiFetch("/api/barangays", { timeout: 30000, retries: 2 });
        const geojson = await res.json();
        const features = geojson.features || [];
        if (!features.length || cancelled) return;
        const props = features[0]?.properties || {};
        const nameCandidates = ["BRGY_NAME","NAME_3","NAME","Barangay","BARANGAY","brgy_name","name","ADM4_EN"];
        const nameCol = nameCandidates.find((k) => props[k] !== undefined) || Object.keys(props)[0];
        const normalised = {
          ...geojson,
          features: features.map((f) => ({
            ...f,
            properties: { ...f.properties, BRGY_NAME: f.properties?.[nameCol] || "Unknown" },
          })),
        };
        setBarangayGeoJSON(normalised);
        const names = normalised.features.map((f) => f.properties.BRGY_NAME).filter((n) => n && n !== "Unknown").sort();
        setBarangayList([...new Set(names)]);
        setBootError("");
      } catch (err) {
        if (!cancelled) {
          setBootError(`Barangay boundaries could not be loaded: ${err.message}`);
        }
      }
    })();

    return () => { cancelled = true; clearTimeout(slowTimer); };
  }, []);

  const rememberScenario = useCallback((duration, depth, antecedent) => {
    setScenarios((prev) => {
      // One entry per distinct scenario, most recent first.
      const rest = prev.filter(
        (e) => !(e.duration === duration && e.depth === depth && e.antecedent === antecedent)
      );
      const next = [{ id: `${duration}_${depth}_${antecedent}_${Date.now()}`,
                      ts: Date.now(), duration, depth, antecedent }, ...rest];
      writeHistory(next);
      return next.slice(0, HISTORY_MAX);
    });
  }, []);

  const clearHistory = () => { setScenarios([]); writeHistory([]); };

  // The per-cell hover readout is a nice-to-have that costs a multi-megabyte
  // download, so it loads in the background after the map is already usable and
  // fails silently. Nothing else on screen depends on it.
  const loadHoverCells = useCallback(async (csvPath) => {
    if (!csvPath) return;
    try {
      const res  = await apiFetch(csvPath, { timeout: 45000 });
      const text = await res.text();
      const lines   = text.trim().split("\n");
      const headers = lines[0].split(",").map((h) => h.trim());
      const xIdx    = headers.indexOf("x_coordinate");
      const yIdx    = headers.indexOf("y_coordinate");
      const codeIdx = headers.indexOf("hazard_code");
      const dIdx    = headers.indexOf("predicted_depth_m");
      const cells   = [];
      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(",");
        if (cols.length < headers.length) continue;
        const [lng, lat] = utmToWgs84Ref(parseFloat(cols[xIdx]), parseFloat(cols[yIdx]));
        cells.push({ lat, lng, code: parseInt(cols[codeIdx]),
          depth: dIdx >= 0 ? parseFloat(cols[dIdx]) : 0 });
      }
      setCsvCells(cells);
    } catch {
      // Hover detail unavailable; the map, legend and summary are unaffected.
    }
  }, []);

  const runPrediction = useCallback(async (dur, dep, ant) => {
    setLoading(true); setError(""); setBootError(""); setResult(null); setCsvCells([]); setTiming(null);
    // Same idea as at startup: say the server is waking rather than showing a
    // spinner that looks identical to a hang.
    const slowTimer = setTimeout(() => setWaking(true), 5000);
    try {
      // The per-barangay summary now arrives inside this one response. It used to
      // be rebuilt in the browser from a 16.6 MB grid plus the prediction CSV, a
      // transfer that regularly died and left the Summary tab empty.
      // Retries matter here specifically because of the cold start. While the
      // instance is booting, the host's proxy answers before the app does, and
      // that response carries no CORS headers, so the browser reports a CORS
      // failure rather than the 502 it actually is. Without a retry that is a
      // dead end: the user clicks Run Prediction, sees "Failed to fetch", and
      // has to click again. These attempts fail fast, so retrying costs little.
      const res = await apiFetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ duration: dur, depth: dep, antecedent: ant }),
        timeout: 90000,
        retries: 2,
        retryDelay: 4000,
      });
      // Server-measured compute time, so it excludes network and cold start.
      // Note the explicit null check: a cache hit reports 0 ms, and `|| null`
      // would treat that as missing and hide the very thing worth showing.
      const rawMs = res.headers.get("X-Prediction-Time-Ms");
      setTiming({
        ms: rawMs === null || rawMs === "" ? null : Number(rawMs),
        cached: res.headers.get("X-Cache") === "HIT",
      });
      const data = await res.json();
      setResult(data); setOverlayVisible(true); setActiveTab("legend");
      setSheetOpen(false); setSidebarCollapsed(false);
      rememberScenario(dur, dep, ant);
      loadHoverCells(data.outputs?.prediction_csv);
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      clearTimeout(slowTimer);
      setWaking(false);
      setLoading(false);
    }
  }, [loadHoverCells, rememberScenario]);

  const handlePredict = (e) => {
    e.preventDefault();
    runPrediction(Number(duration), Number(depth), Number(antecedent));
  };

  // Re-request rather than restoring a stored result: output files are rotated
  // server-side, so a saved result would eventually reference deleted files.
  // The shared cache normally answers this immediately.
  const loadScenario = (s) => {
    setDuration(s.duration); setDepth(s.depth); setAntecedent(s.antecedent);
    setSheetOpen(false);
    runPrediction(s.duration, s.depth, s.antecedent);
  };

  const handleBarangayClick = (name) => {
    if (!barangayGeoJSON) return;
    const feature = barangayGeoJSON.features.find((f) => f.properties?.BRGY_NAME === name);
    if (!feature) return;
    setSelectedBarangay(feature); setHighlightedName(name); setSheetOpen(false);
  };

  const getBarangayStyle = useCallback((feature) =>
    feature.properties?.BRGY_NAME === highlightedName ? barangayHighlightStyle : barangayStyle,
  [highlightedName]);

  // Restyle in place when the selection changes. Previously the layer was given a
  // key of the highlighted name, which tore down and rebuilt all 37 polygons on
  // every click and made the boundaries flicker.
  useEffect(() => {
    const layer = geojsonRef.current;
    if (layer) layer.setStyle(getBarangayStyle);
  }, [highlightedName, getBarangayStyle]);

  const onEachBarangay = useCallback((feature, layer) => {
    const name = feature.properties?.BRGY_NAME || "Unknown";
    layer.bindTooltip(name, { permanent: false, direction: "center", className: "brgy-tooltip" });
    layer.on("click", (e) => {
      setHighlightedName(name);
      setSelectedBarangay(feature);
      // Show popup at click position
      const mapEl = e.originalEvent.target.closest(".leaflet-map") ||
                    document.querySelector(".leaflet-map");
      if (mapEl) {
        const rect = mapEl.getBoundingClientRect();
        setBarangayPopup({
          name,
          x: e.originalEvent.clientX - rect.left,
          y: e.originalEvent.clientY - rect.top,
        });
      }
    });
  }, []);

  const handleTabClick = (tab) => { setActiveTab(tab); setSheetOpen(true); setSidebarCollapsed(false); };

  const summary         = result?.summary;
  const barangaySummary  = result?.barangay_summary || [];
  const areaTotals       = summary?.area_totals;

  const selectedSummary = highlightedName
    ? barangaySummary.find((b) => b.name === highlightedName) || null
    : null;

  // The legend reports one scope at a time: the whole municipality, or a single
  // barangay once one is selected. Both sides use the same denominator rule,
  // every 30 m cell inside the boundary, so the two views are directly comparable
  // and a barangay percentage never has to be mentally rescaled.
  //
  // Percentages are shares of land area. The old denominator was every cell of the
  // raster bounding box, most of which lies outside the municipality, so "No Hazard"
  // was really measuring how much of the rectangle went unmodelled.
  const legendScope = selectedSummary
    ? {
        kind:          "barangay",
        label:         selectedSummary.name,
        areaKm2:       selectedSummary.area_km2,
        counts:        selectedSummary.counts,
        pcts:          selectedSummary.pcts,
        maxDepth:      selectedSummary.max_depth_m,
        maxDepthLand:  selectedSummary.max_depth_land_m,
        channelCells:  selectedSummary.channel_cells,
      }
    : areaTotals
    ? {
        kind:          "municipality",
        label:         "Sipocot",
        areaKm2:       areaTotals.area_km2,
        counts:        areaTotals.class_counts,
        pcts:          areaTotals.class_pcts,
        maxDepth:      summary?.max_depth_m ?? 0,
        maxDepthLand:  summary?.max_depth_land_m ?? 0,
        channelCells:  summary?.n_channel_cells ?? 0,
      }
    : null;

  const withPct = legendScope
    ? [1, 2, 3, 4, 5, 0].map((code) => ({
        code,
        displayName: HAZARD_NAMES[code],
        depthRange:  DEPTH_RANGES[code],
        color:       HAZARD_COLORS[code],
        cell_count:  legendScope.counts?.[String(code)] ?? 0,
        pct:         Number(legendScope.pcts?.[String(code)] ?? 0).toFixed(1),
      }))
    : [];

  const clearSelection = () => { setHighlightedName(""); setSelectedBarangay(null); };

  const filteredBarangays = barangayList.filter((n) =>
    n.toLowerCase().includes(barangaySearch.toLowerCase())
  );

  return (
    <div className="app-root">
      {/* Announced by a screen reader when state changes; visually hidden. */}
      <div className="sr-only" role="status" aria-live="polite">
        {loading
          ? (waking ? "Waking the prediction server." : "Running prediction.")
          : error
          ? `Prediction failed. ${error}`
          : summary && areaTotals
          ? `Prediction complete for ${summary.input_rainfall?.depth} millimetres over `
            + `${summary.input_rainfall?.duration} hours. Across Sipocot, `
            + [5, 4, 3, 2, 1].map((c) => `${HAZARD_NAMES[c]} ${areaTotals.class_pcts?.[String(c)] ?? 0} percent`).join(", ")
            + `, and No Hazard ${areaTotals.class_pcts?.["0"] ?? 0} percent of municipal land area.`
          : ""}
      </div>

      {loading && <LoadingOverlay waking={waking} />}

      {/* ── MOBILE TOP BAR ───────────────────────────────────────────────── */}
      <header className="mobile-topbar">
        <div className="mobile-logo">
          <span className="logo-icon">⛈</span>
          <div><h1>ML-BaHa</h1></div>
        </div>
        <button className="sheet-toggle-btn" onClick={() => setSheetOpen((o) => !o)}
          aria-label={sheetOpen ? "Close panel" : "Open panel"}>
          {sheetOpen ? "✕" : "☰"}
        </button>
      </header>

      {/* ── SIDEBAR ──────────────────────────────────────────────────────── */}
      <aside className={`sidebar ${sheetOpen ? "open" : ""} ${sidebarCollapsed ? "collapsed" : ""}`}>

        <div className="sheet-handle" onClick={() => setSheetOpen((o) => !o)} role="button" aria-label="Toggle panel">
          <div className="sheet-handle-pill" />
          <div className="sheet-peek-row">
            <span className="sheet-peek-label">{TAB_LABELS[activeTab]}</span>
            {highlightedName && <span className="sheet-peek-status">📍 {highlightedName}</span>}
          </div>
        </div>

        <div className="sidebar-header">
          <div className="sidebar-logo">
            <span className="logo-icon">⛈</span>
            <div className="sidebar-title-block">
              <h1>ML-BaHa</h1>
              <p>Machine Learning-Based Flood Hazard Prediction<br/>and Mapping System — Sipocot, Camarines Sur</p>
            </div>
          </div>
        </div>

        <nav className="tab-nav" role="tablist" aria-label="Panel sections">
          {Object.entries(TAB_LABELS).map(([tab, label]) => (
            <button key={tab} id={`tab-${tab}`} role="tab"
              aria-selected={activeTab === tab}
              aria-controls={`panel-${tab}`}
              className={`tab-btn ${activeTab === tab ? "active" : ""}`}
              onClick={() => handleTabClick(tab)}>{label}</button>
          ))}
        </nav>

        <div className="tab-content">

          {/* ── INPUTS ──────────────────────────────────────────────────── */}
          {activeTab === "inputs" && (
            <div className="tab-pane" role="tabpanel" id="panel-inputs" aria-labelledby="tab-inputs" tabIndex={-1}>
              <div className="section-label">Rainfall Parameters</div>
              <form onSubmit={handlePredict} className="input-form">
                <div className="input-group">
                  <label htmlFor="duration">Storm Duration <span className="unit">hours</span></label>
                  <input id="duration" type="number" inputMode="decimal"
                    value={duration} onChange={(e) => setDuration(e.target.value)} min="0" step="0.1" required />
                </div>
                <div className="input-group">
                  <label htmlFor="depth">Rainfall Depth <span className="unit">mm</span></label>
                  <input id="depth" type="number" inputMode="decimal"
                    value={depth} onChange={(e) => setDepth(e.target.value)} min="0" step="0.1" required />
                </div>
                <div className="input-group">
                  <label htmlFor="antecedent">Antecedent Rainfall <span className="unit">mm</span></label>
                  <input id="antecedent" type="number" inputMode="decimal"
                    value={antecedent} onChange={(e) => setAntecedent(e.target.value)} min="0" step="0.1" required />
                </div>
                <button type="submit" className="predict-btn" disabled={loading}>
                  {loading ? <><span className="btn-spinner" /> Running…</> : "▶ Run Prediction"}
                </button>
              </form>

              {error && <div className="error-box">{error}</div>}
              {!error && bootError && <div className="error-box">{bootError}</div>}

              {barangayList.length > 0 && (
                <div className="barangay-section">
                  <div className="section-label">
                    Barangays <span className="section-count">{barangayList.length}</span>
                  </div>
                  <div className="brgy-search-wrap">
                    <span className="brgy-search-icon">🔍</span>
                    <input className="brgy-search" type="text" placeholder="Search barangay…"
                      value={barangaySearch} onChange={(e) => setBarangaySearch(e.target.value)} />
                    {barangaySearch && (
                      <button className="brgy-search-clear" onClick={() => setBarangaySearch("")}>✕</button>
                    )}
                  </div>
                  <div className="brgy-list">
                    {filteredBarangays.length === 0
                      ? <p className="empty-hint">No match for "{barangaySearch}"</p>
                      : filteredBarangays.map((name) => (
                        <button key={name}
                          className={`brgy-item ${highlightedName === name ? "active" : ""}`}
                          onClick={() => handleBarangayClick(name)}>
                          <span className="brgy-pin">{highlightedName === name ? "📍" : "▸"}</span>
                          {name}
                        </button>
                      ))
                    }
                  </div>
                </div>
              )}

              {barangayList.length === 0 && !bootError && (
                <p className="empty-hint" style={{ marginTop: 8 }}>
                  Loading barangay boundaries…
                </p>
              )}
            </div>
          )}

          {/* ── LEGEND ──────────────────────────────────────────────────── */}
          {activeTab === "legend" && (
            <div className="tab-pane" role="tabpanel" id="panel-legend" aria-labelledby="tab-legend" tabIndex={-1}>
              {!result ? (
                <p className="empty-hint">Run a prediction first to see the hazard legend.</p>
              ) : (
                <>
                  <div className="legend-scope">
                    <div className="legend-scope-head">
                      <div>
                        <span className="legend-scope-kind">
                          {legendScope?.kind === "barangay" ? "Barangay" : "Municipality"}
                        </span>
                        <h3 className="legend-scope-name">{legendScope?.label ?? "Sipocot"}</h3>
                      </div>
                      <span className="legend-scope-area">{legendScope?.areaKm2 ?? 0} km²</span>
                    </div>
                    {timing?.ms != null && (
                      <p className="legend-scope-timing">
                        {timing.cached
                          ? "Served from cache, no recomputation"
                          : `Computed in ${(timing.ms / 1000).toFixed(1)} s`}
                      </p>
                    )}
                    {legendScope?.kind === "barangay" ? (
                      <button className="legend-scope-back" onClick={clearSelection}>
                        ← Back to all of Sipocot
                      </button>
                    ) : (
                      <p className="legend-scope-hint">
                        Click any barangay on the map or in the list to see its own breakdown.
                      </p>
                    )}
                  </div>

                  <div className="legend-list">
                    <h3 className="legend-title">
                      Hazard classes · percent of {legendScope?.kind === "barangay" ? "barangay" : "municipal"} area
                    </h3>
                    {(() => {
                      const maxP = Math.max(...withPct.map((c) => parseFloat(c.pct)));
                      return withPct.map((cls) => (
                        <div className="legend-item" key={cls.code}>
                          <HazardSwatch color={cls.color} />
                          <div className="legend-text">
                            <span className="legend-name">{cls.displayName}</span>
                            <span className="legend-depth">{cls.depthRange}</span>
                          </div>
                          <div className="legend-pct-block">
                            <span className="legend-pct">{cls.pct}%</span>
                            <div className="legend-bar-track">
                              <div className="legend-bar-fill"
                                style={{
                                  width: `${(parseFloat(cls.pct) / maxP) * 100}%`,
                                  background: cls.color || "#888"
                                }} />
                            </div>
                          </div>
                        </div>
                      ));
                    })()}
                  </div>

                  {legendScope && legendScope.maxDepth > 0 && (
                    <div className="legend-depth-note">
                      <span>
                        Max depth on land <strong>{legendScope.maxDepthLand.toFixed(2)} m</strong>
                      </span>
                      {legendScope.channelCells > 0 && (
                        <span className="legend-depth-channel">
                          {legendScope.maxDepth.toFixed(2)} m including river channel cells
                        </span>
                      )}
                    </div>
                  )}

                  {highlightedName && !selectedSummary && (
                    <p className="empty-hint" style={{ marginTop: 10 }}>
                      No summary recorded for {highlightedName} in this scenario.
                    </p>
                  )}

                  <div className="map-controls">
                    <div className="control-row">
                      <label className="control-label">
                        <input type="checkbox" checked={overlayVisible}
                          onChange={(e) => setOverlayVisible(e.target.checked)} />
                        Show hazard overlay
                      </label>
                    </div>
                    <div className="control-row">
                      <label className="control-label">
                        Opacity: <strong>{Math.round(opacity * 100)}%</strong>
                      </label>
                      <input type="range" min="0.1" max="1" step="0.05" value={opacity}
                        onChange={(e) => setOpacity(Number(e.target.value))}
                        className="opacity-slider" disabled={!overlayVisible} />
                    </div>
                    {barangayGeoJSON && (
                      <div className="control-row">
                        <label className="control-label">
                          <input type="checkbox" checked={barangayVisible}
                            onChange={(e) => setBarangayVisible(e.target.checked)} />
                          Show barangay boundaries
                        </label>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── BARANGAY SUMMARY ────────────────────────────────────────── */}
          {activeTab === "summary" && (
            <div className="tab-pane" role="tabpanel" id="panel-summary" aria-labelledby="tab-summary" tabIndex={-1}>
              {!result ? (
                <p className="empty-hint">Run a prediction first to see the barangay hazard summary.</p>
              ) : barangaySummary.length === 0 ? (
                <p className="empty-hint">
                  This scenario carries no barangay summary. Re-run the prediction to generate one.
                </p>
              ) : (
                <>
                  <div className="section-label" style={{ marginBottom: 6 }}>
                    Hazard by Barangay
                    <span className="section-count">{barangaySummary.length}</span>
                  </div>

                  {/* Search */}
                  <div className="brgy-search-wrap" style={{ marginBottom: 8 }}>
                    <span className="brgy-search-icon">🔍</span>
                    <input className="brgy-search" type="text"
                      placeholder="Search barangay…"
                      value={summarySearch}
                      onChange={(e) => setSummarySearch(e.target.value)} />
                    {summarySearch && (
                      <button className="brgy-search-clear"
                        onClick={() => setSummarySearch("")}>✕</button>
                    )}
                  </div>

                  {/* Table */}
                  <div className="brgy-summary-table-wrap">
                    <table className="brgy-summary-table">
                      <thead>
                        <tr>
                          <th>Barangay</th>
                          <th>Dominant</th>
                          <th title="No Hazard">NH %</th>
                          <th title="Low">L %</th>
                          <th title="Moderate">Mo %</th>
                          <th title="High">H %</th>
                          <th title="Very High">VH %</th>
                          <th title="Extreme">X %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {barangaySummary
                          .filter((b) => b.name.toLowerCase().includes(summarySearch.toLowerCase()))
                          .map((b) => (
                            <tr key={b.name}
                              className={highlightedName === b.name ? "active" : ""}
                              tabIndex={0}
                              role="button"
                              aria-label={`${b.name}, dominant hazard ${HAZARD_NAMES[b.dominant]}, `
                                + `${b.classified_pct}% of its area hazard-classified. `
                                + `Select to show this barangay in the legend.`}
                              onClick={() => handleBarangayClick(b.name)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter" || e.key === " ") {
                                  e.preventDefault();
                                  handleBarangayClick(b.name);
                                }
                              }}
                              style={{ cursor: "pointer" }}>
                              <td className="brgy-summary-name">{b.name}</td>
                              <td>
                                <span className="brgy-hazard-badge"
                                  style={{ background: HAZARD_COLORS[b.dominant],
                                           color: HAZARD_ON_COLOR[b.dominant] }}>
                                  {HAZARD_NAMES[b.dominant] ?? "—"}
                                </span>
                              </td>
                              {/* NH (code 0) first, then Low–Extreme (codes 1–5) */}
                              {[0,1,2,3,4,5].map((c) => (
                                <td key={c} className="brgy-summary-count">
                                  {parseFloat(b.pcts?.[c] || 0) > 0
                                    ? <span style={{ fontWeight: c === b.dominant ? 700 : 500 }}>
                                        {b.pcts[c]}%
                                      </span>
                                    : <span className="muted" aria-label="none">—</span>
                                  }
                                </td>
                              ))}
                            </tr>
                          ))
                        }
                      </tbody>
                    </table>
                  </div>

                  {/* Legend for column abbreviations */}
                  <div className="summary-legend-hint">
                    NH = No Hazard &nbsp;·&nbsp; L = Low &nbsp;·&nbsp; Mo = Moderate &nbsp;·&nbsp;
                    H = High &nbsp;·&nbsp; VH = Very High &nbsp;·&nbsp; X = Extreme
                    <br />
                    Values represent percent (%) of total barangay area.
                    <br />
                    Select a barangay for its predicted depths, reported separately for land and for
                    river channel cells.
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── HISTORY ─────────────────────────────────────────────────── */}
          {activeTab === "history" && (
            <div className="tab-pane" role="tabpanel" id="panel-history" aria-labelledby="tab-history" tabIndex={-1}>
              <div className="history-header">
                <h3>Your Scenarios</h3>
                {scenarios.length > 0 && (
                  <button className="refresh-btn" onClick={clearHistory}>✕ Clear</button>
                )}
              </div>
              {scenarios.length === 0
                ? <p className="empty-hint">No scenarios yet. Run a prediction to start building history.</p>
                : <div className="scenario-list">
                    {scenarios.map((s) => <ScenarioRow key={s.id} s={s} onLoad={loadScenario} />)}
                  </div>
              }
            </div>
          )}

          {/* ── DOWNLOADS / EXPORT ──────────────────────────────────────── */}
          {activeTab === "downloads" && (
            <div className="tab-pane" role="tabpanel" id="panel-downloads" aria-labelledby="tab-downloads" tabIndex={-1}>
              {!result ? (
                <p className="empty-hint">Run a prediction to enable downloads.</p>
              ) : (
                <div className="download-list">
                  <p className="dl-section-label">Raster outputs</p>
                  <DownloadBtn href={outputUrls.depthRaster}   label="Depth Raster (.tif)"  icon="📐" />
                  <DownloadBtn href={outputUrls.hazardRaster}  label="Hazard Raster (.tif)" icon="🗺" />
                  <p className="dl-section-label">Tabular outputs</p>
                  <DownloadBtn href={outputUrls.predictionCsv} label="Prediction CSV"        icon="📊" />
                  <DownloadBtn href={outputUrls.summaryJson}   label="Summary JSON"          icon="📋" />
                </div>
              )}
            </div>
          )}

          {/* ── ABOUT ───────────────────────────────────────────────────── */}
          {activeTab === "about" && (
            <div className="tab-pane" role="tabpanel" id="panel-about" aria-labelledby="tab-about" tabIndex={-1}>

              {/* System overview */}
              <div className="about-card">
                <div className="about-card-header">
                  <span className="about-card-icon">⛈</span>
                  <span className="about-card-title">ML-BaHa</span>
                </div>
                <p className="about-card-body">
                  ML-BaHa is a Machine Learning-Based Flood Hazard Prediction and Mapping System
                  developed for Sipocot, Camarines Sur. It uses a trained Random Forest model to predict
                  flood inundation depth across a spatial grid from rainfall inputs, then classifies
                  each grid cell into a standardized hazard tier.
                </p>
              </div>

              {/* Model */}
              <div className="about-section-label">Prediction Model</div>
              <div className="about-card">
                <div className="about-row">
                  <span className="about-row-key">Algorithm</span>
                  <span className="about-row-val">Random Forest (Ensemble of Decision Trees)</span>
                </div>
                <div className="about-row">
                  <span className="about-row-key">Model type</span>
                  <span className="about-row-val">Regression (flood depth, metres)</span>
                </div>
                <div className="about-row">
                  <span className="about-row-key">Output</span>
                  <span className="about-row-val">Flood depth per grid cell → hazard class</span>
                </div>
                <div className="about-row">
                  <span className="about-row-key">Tuning</span>
                  <span className="about-row-val">Randomized search (30 trials, stratified)</span>
                </div>
                <div className="about-row">
                  <span className="about-row-key">Format</span>
                  <span className="about-row-val">Scikit-learn compatible (.pkl)</span>
                </div>
              </div>

              {/* Input features */}
              <div className="about-section-label">Input Features (12 total)</div>
              <div className="about-card">
                <p className="about-sub-label">Dynamic (per prediction — 6 features)</p>
                {[
                  ["Duration",         "Storm duration in hours"],
                  ["Depth",            "Total rainfall depth in mm"],
                  ["Antecedent",       "Prior rainfall in mm"],
                  ["Intensity",        "Depth ÷ Duration (mm/hr)"],
                  ["Total Rain",       "Depth + Antecedent (mm)"],
                  ["Antecedent Ratio", "Antecedent ÷ Depth"],
                ].map(([k, v]) => (
                  <div className="about-row" key={k}>
                    <span className="about-row-key">{k}</span>
                    <span className="about-row-val">{v}</span>
                  </div>
                ))}
                <p className="about-sub-label" style={{ marginTop: 10 }}>Static (precomputed per grid cell — 6 features)</p>
                {[
                  ["X Coordinate",    "Easting (UTM grid centroid)"],
                  ["Y Coordinate",    "Northing (UTM grid centroid)"],
                  ["Elevation",       "LiDAR-IfSAR merged DTM (m)"],
                  ["Slope",           "Terrain slope derived from DTM (degrees)"],
                  ["Log₁₀ Flow Acc.", "log₁₀ of flow accumulation (drainage proxy)"],
                  ["TWI",             "Topographic Wetness Index (ln(A / tan β))"],
                ].map(([k, v]) => (
                  <div className="about-row" key={k}>
                    <span className="about-row-key">{k}</span>
                    <span className="about-row-val">{v}</span>
                  </div>
                ))}
              </div>

              {/* Hazard classification */}
              <div className="about-section-label">Hazard Classification</div>
              <div className="about-card">
                <p className="about-card-body" style={{ marginBottom: 10 }}>
                  Predicted flood depths are classified into five hazard tiers following
                  the depth thresholds established in Philippine flood risk assessment
                  literature. The classification scheme is consistent with the
                  DENR-MGB Flood Susceptibility Maps (Eusebio et al., 2022) and
                  the UP NOAH / Project NOAH flood hazard framework (Lagmay et al., 2017),
                  which define hazard levels based on inundation depth relative to
                  human body height and structural damage thresholds.
                </p>
                {[
                  ["Low",       "0.0 – 0.5 m",  "#facc15", "Ankle- to knee-level. Minimal structural risk; negligible damage probability (Besarra et al., 2025)."],
                  ["Moderate",  "0.5 – 1.0 m",  "#f97316", "Knee- to waist-level. Lower bound of medium hazard per UP NOAH (2024); ~10% minor damage probability."],
                  ["High",      "1.0 – 1.5 m",  "#ea580c", "Waist- to chest-level. MGB 'High' susceptibility threshold; significant inundation risk."],
                  ["Very High", "1.5 – 2.0 m",  "#dc2626", "Chest- to neck-level. 87% probability of minor structural damage (Besarra et al., 2025)."],
                  ["Extreme",   "> 2.0 m",       "#991b1b", "Above head level. Near-total structural damage expected; immediate evacuation required."],
                  ["No Hazard", "≤ 0 m / dry",  "#374151", "No predicted inundation under the given rainfall scenario."],
                ].map(([name, range, color, note]) => (
                  <div key={name} style={{ borderBottom: "1px solid var(--navy-700)", padding: "8px 0" }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 3 }}>
                      <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
                        <span style={{ width: 11, height: 11, borderRadius: 3, background: color, flexShrink: 0, display: "inline-block", border: "1px solid rgba(255,255,255,0.15)" }} />
                        <span style={{ fontSize: 12, fontWeight: 700, color: "var(--slate-200)" }}>{name}</span>
                      </span>
                      <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--cyan-300)" }}>{range}</span>
                    </div>
                    <p style={{ fontSize: 10, color: "var(--slate-500)", lineHeight: 1.6, paddingLeft: 18, margin: 0 }}>{note}</p>
                  </div>
                ))}
                <p style={{ fontSize: 9.5, color: "var(--slate-500)", marginTop: 10, lineHeight: 1.7 }}>
                  <strong style={{ color: "var(--slate-400)" }}>References:</strong> Eusebio et al. (2022) <em>Appl. Sci.</em> 12(19), 9456 — MGB-based 5-class scheme, Romblon PH.
                  Lagmay et al. (2017) <em>J. Environ. Sci.</em> 59, 13–23 — UP NOAH/Project NOAH PH framework.
                  Besarra et al. (2025) <em>J. Flood Risk Mgmt.</em> — fragility functions, Leyte PH.
                </p>
              </div>

              {/* Reading the depths */}
              <div className="about-section-label">Reading the Predicted Depths</div>
              <div className="about-card">
                <p className="about-card-body">
                  The model predicts a water depth for every 30 m cell, including cells that lie on
                  the river network itself. In those channel cells the predicted value is the depth
                  of water in the channel, not the depth of flooding on land, so during a large storm
                  they can legitimately reach well beyond the Extreme threshold. They are counted in
                  the hazard classes like any other cell and nothing is removed, but the barangay
                  table reports a separate <strong>Max land</strong> figure that leaves them out, and
                  that is the number to use when judging inundation in populated areas.
                </p>
                <p className="about-card-body" style={{ marginTop: 8 }}>
                  A cell is treated as river channel when its contributing drainage area reaches
                  10<sup>4</sup> cells, about 9 km², a conventional channel-initiation threshold.
                </p>
              </div>

              {/* Coverage */}
              <div className="about-section-label">What the Percentages Mean</div>
              <div className="about-card">
                <p className="about-card-body">
                  Hazard percentages in the legend and the barangay table are shares of land area,
                  measured against every 30 m cell inside the relevant boundary. Cells the model
                  filtered out as permanently dry are included in that denominator and counted as
                  No Hazard, so the classes always sum to 100% of the area rather than to the subset
                  of cells the model evaluated.
                </p>
              </div>

              {/* Study area */}
              <div className="about-section-label">Study Area</div>
              <div className="about-card">
                {[
                  ["Municipality", "Sipocot"],
                  ["Province",     "Camarines Sur"],
                  ["Region",       "Bicol Region (Region V)"],
                  ["Country",      "Philippines"],
                ].map(([k, v]) => (
                  <div className="about-row" key={k}>
                    <span className="about-row-key">{k}</span>
                    <span className="about-row-val">{v}</span>
                  </div>
                ))}
              </div>

              {/* Data sources */}
              <div className="about-section-label">Data Sources</div>
              <div className="about-card">
                {[
                  ["DTM",             "LiDAR-IfSAR merged Digital Terrain Model"],
                  ["Slope",           "Derived from DTM"],
                  ["Flow Accumulation","Derived from DTM (hydrological routing)"],
                  ["TWI",             "Derived from slope and flow accumulation"],
                  ["Barangay bounds", "NAMRIA / PhilGIS shapefile"],
                  ["Rainfall data",   "PAGASA historical records"],
                  ["Flood records",   "NDRRMC / LGU flood reports"],
                ].map(([k, v]) => (
                  <div className="about-row" key={k}>
                    <span className="about-row-key">{k}</span>
                    <span className="about-row-val">{v}</span>
                  </div>
                ))}
              </div>

              {/* Tech stack */}
              <div className="about-section-label">Technology Stack</div>
              <div className="about-card">
                <p className="about-sub-label">Backend</p>
                {[
                  ["Framework",    "FastAPI (Python)"],
                  ["ML Library",   "scikit-learn (Random Forest)"],
                  ["Raster I/O",   "rasterio / numpy / scipy"],
                  ["Serving",      "Uvicorn ASGI server"],
                ].map(([k, v]) => (
                  <div className="about-row" key={k}>
                    <span className="about-row-key">{k}</span>
                    <span className="about-row-val">{v}</span>
                  </div>
                ))}
                <p className="about-sub-label" style={{ marginTop: 10 }}>Frontend</p>
                {[
                  ["Framework",   "React + Vite"],
                  ["Map library", "Leaflet / react-leaflet"],
                  ["Styling",     "Custom CSS (IBM Plex Sans)"],
                ].map(([k, v]) => (
                  <div className="about-row" key={k}>
                    <span className="about-row-key">{k}</span>
                    <span className="about-row-val">{v}</span>
                  </div>
                ))}
              </div>

              {/* Footer */}
              <div className="about-footer">
                <span>ML-BaHa © {new Date().getFullYear()}</span>
                <span>Sipocot, Camarines Sur, Philippines</span>
              </div>

            </div>
          )}

        </div>{/* end tab-content */}

        {/* ── HOTLINES FOOTER ───────────────────────────────────────────── */}
        <div className="hotlines-footer">
          <div className="hotlines-header">
            <span className="hotlines-icon">🚨</span>
            <span className="hotlines-title">Emergency Hotlines — Sipocot, CamSur</span>
          </div>
          <div className="hotlines-grid">
            {HOTLINES.map((h) => (
              <a key={h.label} href={`tel:${h.number.replace(/[^0-9]/g,"")}`} className="hotline-card">
                <span className="hotline-icon">{h.icon}</span>
                <div className="hotline-info">
                  <span className="hotline-label">{h.label}</span>
                  <span className="hotline-number">{h.number}</span>
                </div>
              </a>
            ))}
          </div>
        </div>
      </aside>

      {/* ── MAP ──────────────────────────────────────────────────────────── */}
      <main className="map-area">
        <button className="sidebar-collapse-btn"
          onClick={() => setSidebarCollapsed((c) => !c)}
          aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {sidebarCollapsed ? "▶" : "◀"}
        </button>

        {!result && (
          <div className="map-placeholder">
            <div className="placeholder-inner">
              <span className="placeholder-icon">⛈</span>
              <h2>No prediction yet</h2>
              <p>Enter rainfall parameters and click <strong>Run Prediction</strong> to generate the ML-BaHa flood hazard map.</p>
            </div>
          </div>
        )}

        <MapContainer center={[13.78, 123.0]} zoom={11} scrollWheelZoom
          className="leaflet-map">
          {/* The raster conveys its content through colour alone, so the figures
              are also published as text in the live region above and in the
              barangay table, which is keyboard navigable. */}
          {bounds && <MapFitter bounds={bounds} />}
          {selectedBarangay && <BarangayFocuser feature={selectedBarangay} />}

          <LayersControl position="topright">
            <LayersControl.BaseLayer checked name="OpenStreetMap">
              <TileLayer attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
            </LayersControl.BaseLayer>
            <LayersControl.BaseLayer name="Satellite">
              <TileLayer attribution="&copy; Esri"
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                maxZoom={19} />
            </LayersControl.BaseLayer>
            {/* Esri's topographic service, same provider as the imagery layer above.
                The previous source was mt{0-3}.google.com/vt, an undocumented endpoint
                that Google does not support for third-party use and can block without
                notice, which would leave this layer blank. */}
            <LayersControl.BaseLayer name="Terrain">
              <TileLayer attribution="Tiles &copy; Esri"
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
                maxZoom={19} />
            </LayersControl.BaseLayer>
            {overlayUrl && bounds && (
              <LayersControl.Overlay checked={overlayVisible} name="Flood Hazard Overlay">
                <ImageOverlay url={overlayUrl} bounds={bounds} opacity={overlayVisible ? opacity : 0} />
              </LayersControl.Overlay>
            )}
            {barangayGeoJSON && barangayVisible && (
              <LayersControl.Overlay checked name="Barangay Boundaries">
                <GeoJSON data={barangayGeoJSON}
                  style={getBarangayStyle} onEachFeature={onEachBarangay} ref={geojsonRef} />
              </LayersControl.Overlay>
            )}
          </LayersControl>

          <ScaleControl position="bottomleft" />
          {overlayUrl && bounds && (
            <MapHoverTooltip overlayBounds={bounds} csvCells={csvCells} />
          )}
          <MapClickDismiss onDismiss={() => setBarangayPopup(null)} />
        </MapContainer>

        {/* ── MAP LEGEND OVERLAY — bottom right ────────────────────────── */}
        {result && withPct.length > 0 && (
          <div className="map-legend-overlay">
            <div className="map-legend-title">
              {legendScope?.kind === "barangay" ? legendScope.label : "Flood Hazard"}
            </div>
            <div className="map-legend-sub">
              {legendScope?.kind === "barangay"
                ? `Barangay · ${legendScope.areaKm2} km²`
                : `Sipocot · ${legendScope?.areaKm2 ?? 0} km²`}
            </div>
            {withPct.map((cls) => (
              <div className="map-legend-row" key={cls.code}>
                <span className="map-legend-swatch" style={{ background: cls.color || "#888" }} />
                <span className="map-legend-name">{cls.displayName}</span>
                <span className="map-legend-pct">{cls.pct}%</span>
              </div>
            ))}
            {/* Depth color scale */}
            <div className="depth-scale-title">Flood Depth (m)</div>
            <div className="depth-scale-bar" />
            <div className="depth-scale-labels">
              <span>0</span><span>0.5</span><span>1.0</span>
              <span>1.5</span><span>2.0</span><span>2.0+</span>
            </div>
          </div>
        )}

        {/* ── BARANGAY POPUP ───────────────────────────────────────────── */}
        {barangayPopup && (() => {
          const bData = barangaySummary.find((b) => b.name === barangayPopup.name);
          return (
            <div className="brgy-popup"
              style={{ left: barangayPopup.x + 12, top: barangayPopup.y - 12 }}>
              <div className="brgy-popup-header">
                <span className="brgy-popup-name">{barangayPopup.name}</span>
                <button className="brgy-popup-close"
                  onClick={() => setBarangayPopup(null)}>✕</button>
              </div>
              {bData ? (
                <div className="brgy-popup-body">
                  <div className="brgy-popup-dominant">
                    <span className="brgy-popup-dom-label">Dominant Hazard</span>
                    <span className="brgy-hazard-badge"
                      style={{ background: HAZARD_COLORS[bData.dominant],
                               color: HAZARD_ON_COLOR[bData.dominant] }}>
                      {HAZARD_NAMES[bData.dominant]}
                    </span>
                  </div>
                  <div className="brgy-popup-grid">
                    {[0,1,2,3,4,5].map((c) => (
                      parseFloat(bData.pcts?.[c] || 0) > 0 && (
                        <div key={c} className="brgy-popup-cell">
                          <span className="brgy-popup-cell-dot"
                            style={{ background: HAZARD_COLORS[c] }} />
                          <span className="brgy-popup-cell-name">
                            {c === 0 ? "No Hazard" : HAZARD_NAMES[c]}
                          </span>
                          <span className="brgy-popup-cell-pct"
                            style={{ color: HAZARD_COLORS[c] }}>
                            {bData.pcts[c]}%
                          </span>
                        </div>
                      )
                    ))}
                  </div>
                  {bData.max_depth_m > 0 && (
                    <div className="brgy-popup-depth">
                      Max depth on land: <strong>{bData.max_depth_land_m.toFixed(2)} m</strong>
                      {bData.channel_cells > 0 && (
                        <span style={{ display: "block", fontSize: 10, opacity: 0.7, marginTop: 2 }}>
                          {bData.max_depth_m.toFixed(2)} m including river channel cells
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="brgy-popup-body">
                  <p style={{ fontSize: 12, color: "var(--slate-400)" }}>
                    Run a prediction to see hazard breakdown.
                  </p>
                </div>
              )}
            </div>
          );
        })()}
      </main>

      <div className={`sheet-backdrop ${sheetOpen ? "visible" : ""}`}
        onClick={() => setSheetOpen(false)} aria-hidden="true" />
    </div>
  );
}
