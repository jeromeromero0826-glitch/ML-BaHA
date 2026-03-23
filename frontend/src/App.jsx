import { useState, useEffect, useCallback, useRef } from "react";
import {
  MapContainer, TileLayer, ImageOverlay,
  LayersControl, ScaleControl, GeoJSON, useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// ── Point-in-polygon (ray casting) ───────────────────────────────────────────
function pointInPolygon(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1];
    const xj = ring[j][0], yj = ring[j][1];
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function pointInFeature(x, y, feature) {
  const geom = feature.geometry;
  if (!geom) return false;
  const polys = geom.type === "Polygon"
    ? [geom.coordinates]
    : geom.type === "MultiPolygon"
    ? geom.coordinates
    : [];
  for (const poly of polys) {
    if (pointInPolygon(x, y, poly[0])) return true;
  }
  return false;
}

// Compute bounding box for a feature (for fast pre-filter)
function featureBBox(feature) {
  const geom = feature.geometry;
  const rings = geom.type === "Polygon"
    ? [geom.coordinates[0]]
    : geom.type === "MultiPolygon"
    ? geom.coordinates.map((p) => p[0])
    : [];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const ring of rings) {
    for (const [cx, cy] of ring) {
      if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
    }
  }
  return { minX, maxX, minY, maxY };
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

  useEffect(() => {
    if (!overlayBounds || !csvCells?.length) {
      setTooltip(null);
      return;
    }

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
    return () => { map.off("mousemove", onMove); map.off("mouseout", onOut); };
  }, [map, overlayBounds, csvCells]);

  if (!tooltip) return null;
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
  const r = s.rainfall;
  return (
    <div className="scenario-row">
      <div className="scenario-meta">
        <span className="scenario-ts">{s.timestamp}</span>
        <span className="scenario-params">{r.depth}mm / {r.duration}h / API:{r.antecedent}mm</span>
      </div>
      <button className="scenario-load-btn" onClick={() => onLoad(s)}>Load</button>
    </div>
  );
}

function LoadingOverlay() {
  return (
    <div className="loading-overlay" aria-live="polite">
      <div className="spinner-ring" />
      <p>Running ML-BaHa prediction…</p>
      <p className="loading-sub">Computing flood depth for all grid cells</p>
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
  const [scenarios, setScenarios] = useState([]);

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
  const [barangaySummary,  setBarangaySummary]  = useState([]);   // per-barangay hazard counts
  const [summaryLoading,   setSummaryLoading]   = useState(false);
  const [summarySearch,    setSummarySearch]    = useState("");
  const [barangayPopup,    setBarangayPopup]    = useState(null); // { name, x, y }

  const geojsonRef  = useRef(null);
  const [csvCells, setCsvCells] = useState([]); // lat/lng cells for hover tooltip
  const [gridTotal, setGridTotal] = useState(0); // total cells for legend %

  const mapOutputs = result?.map_outputs;
  const overlayUrl = mapOutputs?.hazard_png ? `${API_BASE_URL}${mapOutputs.hazard_png}` : null;
  const bounds     = mapOutputs?.bounds || null;

  const outputUrls = {
    depthRaster:   result?.outputs?.depth_raster   ? `${API_BASE_URL}${result.outputs.depth_raster}`   : null,
    hazardRaster:  result?.outputs?.hazard_raster  ? `${API_BASE_URL}${result.outputs.hazard_raster}`  : null,
    predictionCsv: result?.outputs?.prediction_csv ? `${API_BASE_URL}${result.outputs.prediction_csv}` : null,
    summaryJson:   result?.outputs?.summary_json   ? `${API_BASE_URL}${result.outputs.summary_json}`   : null,
  };

  // ── Fetch barangays ──────────────────────────────────────────────────────
  useEffect(() => {
    // Fetch grid total for legend percentage denominator
    fetch(`${API_BASE_URL}/api/grid-total`)
      .then((r) => r.ok ? r.json() : null)
      .then((d) => { if (d?.total_cells) setGridTotal(d.total_cells); })
      .catch(() => {});
    fetch(`${API_BASE_URL}/api/barangays`)
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((geojson) => {
        const features = geojson.features || [];
        if (!features.length) return;
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
      })
      .catch((e) => console.warn("[barangays]", e));
  }, []);

  // ── Barangay hazard summary ──────────────────────────────────────────────
  const computeBarangaySummary = useCallback(async (csvUrl, geojson) => {
    if (!csvUrl || !geojson?.features?.length) return;
    setSummaryLoading(true);
    try {
      // Fetch prediction CSV (active cells with hazard codes)
      const [predRes, allRes] = await Promise.all([
        fetch(csvUrl),
        fetch(`${API_BASE_URL}/api/grid-all-cells`),
      ]);
      const predText = await predRes.text();
      const allText  = await allRes.text();

      // Parse prediction CSV → map of "row_col" → {code, depth}
      const predLines  = predText.trim().split("\n");
      const predHdrs   = predLines[0].split(",").map((h) => h.trim());
      const pRowIdx    = predHdrs.indexOf("row");
      const pColIdx    = predHdrs.indexOf("col");
      const pCodeIdx   = predHdrs.indexOf("hazard_code");
      const pDepthIdx  = predHdrs.indexOf("predicted_depth_m");
      const predMap    = new Map();
      for (let i = 1; i < predLines.length; i++) {
        const cols = predLines[i].split(",");
        if (cols.length < predHdrs.length) continue;
        const key = `${cols[pRowIdx]}_${cols[pColIdx]}`;
        predMap.set(key, {
          code:  parseInt(cols[pCodeIdx]),
          depth: pDepthIdx >= 0 ? parseFloat(cols[pDepthIdx]) : 0,
        });
      }

      // Parse all-cells CSV → full spatial grid
      const allLines = allText.trim().split("\n");
      const allHdrs  = allLines[0].split(",").map((h) => h.trim());
      const aRowIdx  = allHdrs.indexOf("row");
      const aColIdx  = allHdrs.indexOf("col");
      const aXIdx    = allHdrs.indexOf("x_coordinate");
      const aYIdx    = allHdrs.indexOf("y_coordinate");

      const allCells = [];
      for (let i = 1; i < allLines.length; i++) {
        const cols = allLines[i].split(",");
        if (cols.length < allHdrs.length) continue;
        const row = cols[aRowIdx], col = cols[aColIdx];
        const key = `${row}_${col}`;
        const pred = predMap.get(key);
        const utmX = parseFloat(cols[aXIdx]);
        const utmY = parseFloat(cols[aYIdx]);
        const [lng, lat] = utmToWgs84Ref(utmX, utmY);
        allCells.push({
          x:     lng,
          y:     lat,
          code:  pred ? pred.code  : 0,  // filtered-out = No Hazard
          depth: pred ? pred.depth : 0,
        });
      }

      // Pre-compute bounding boxes
      const features = geojson.features;
      const bboxes   = features.map((f) => featureBBox(f));

      // Assign each cell to a barangay
      const summary = features.map((f) => ({
        name:       f.properties?.BRGY_NAME || "Unknown",
        counts:     { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 },
        totalCells: 0,
        maxDepth:   0,
        depthSum:   0,
      }));

      for (const cell of allCells) {
        for (let fi = 0; fi < features.length; fi++) {
          const bb = bboxes[fi];
          if (cell.x < bb.minX || cell.x > bb.maxX ||
              cell.y < bb.minY || cell.y > bb.maxY) continue;
          if (pointInFeature(cell.x, cell.y, features[fi])) {
            summary[fi].counts[cell.code]++;
            summary[fi].totalCells++;
            summary[fi].depthSum  += cell.depth;
            if (cell.depth > summary[fi].maxDepth) summary[fi].maxDepth = cell.depth;
            break;
          }
        }
      }

      // Compute percentages and dominant hazard
      const result = summary.map((b) => {
        const total = b.totalCells || 1;
        const pcts  = {};
        for (let c = 0; c <= 5; c++) {
          pcts[c] = b.counts[c] > 0
            ? ((b.counts[c] / total) * 100).toFixed(1)
            : "0";
        }
        // Dominant = hazard class (1-5) with the HIGHEST percentage
        let dominant = 0;
        let maxPct   = 0;
        for (let c = 1; c <= 5; c++) {
          const p = parseFloat(pcts[c] || 0);
          if (p > maxPct) { maxPct = p; dominant = c; }
        }
        return {
          ...b,
          pcts,
          meanDepth: b.totalCells > 0 ? b.depthSum / b.totalCells : 0,
          dominant,
        };
      });

      result.sort((a, b) =>
        b.dominant !== a.dominant
          ? b.dominant - a.dominant
          : a.name.localeCompare(b.name)
      );

      setBarangaySummary(result);
    } catch (e) {
      console.warn("[barangay summary]", e);
    } finally {
      setSummaryLoading(false);
    }
  }, []);

  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/scenarios?limit=20`);
      if (res.ok) setScenarios((await res.json()).scenarios || []);
    } catch (_) {}
  }, []);

  const loadScenario = (s) => {
    setResult({ summary: s.summary, hazard_class_counts: s.summary.hazard_class_counts, outputs: s.outputs, map_outputs: s.map_outputs });
    setActiveTab("legend");
    setSheetOpen(false);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true); setError(""); setResult(null); setCsvCells([]);
    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ duration: Number(duration), depth: Number(depth), antecedent: Number(antecedent) }),
      });
      if (!res.ok) throw new Error((await res.text()) || "Prediction failed.");
      const data = await res.json();
      setResult(data); setOverlayVisible(true); setActiveTab("legend");
      setSheetOpen(false); setSidebarCollapsed(false);
      await fetchHistory();
      // Compute barangay hazard summary from prediction CSV
      if (data.outputs?.prediction_csv && barangayGeoJSON) {
        const csvUrl = `${API_BASE_URL}${data.outputs.prediction_csv}`;
        computeBarangaySummary(csvUrl, barangayGeoJSON);
        // Also build lat/lng cell list for hover tooltip
        fetch(csvUrl).then((r) => r.text()).then((text) => {
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
        }).catch(() => {});
      }
    } catch (err) { setError(err.message || "Failed to fetch"); }
    finally { setLoading(false); }
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

  const onEachBarangay = useCallback((feature, layer) => {
    const name = feature.properties?.BRGY_NAME || "Unknown";
    layer.bindTooltip(name, { permanent: false, direction: "center", className: "brgy-tooltip" });
    layer.on("click", (e) => {
      setHighlightedName(name);
      setSelectedBarangay(feature);
      // Auto-compute summary if prediction exists but summary not yet loaded
      if (barangaySummary.length === 0) {
        const csvPath = result?.outputs?.prediction_csv;
        if (csvPath && barangayGeoJSON) {
          computeBarangaySummary(`${API_BASE_URL}${csvPath}`, barangayGeoJSON);
        }
      }
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
  }, [barangaySummary, result, barangayGeoJSON, computeBarangaySummary]);

  const handleTabClick = (tab) => { setActiveTab(tab); setSheetOpen(true); setSidebarCollapsed(false); };

  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const summary       = result?.summary;
  const hazardClasses = result?.hazard_class_counts || [];
  const totalCells    = summary?.n_cells || 1;
  const legendTotal   = gridTotal > 0 ? gridTotal : totalCells;

  // Only codes 1-5 from model (exclude code 0 — will be recomputed below)
  const activePct = hazardClasses
    .filter((cls) => cls.code >= 1)
    .map((cls) => ({
      ...cls,
      displayName: HAZARD_NAMES[cls.code] ?? cls.name,
      depthRange:  DEPTH_RANGES[cls.code] ?? "—",
      pct:         ((cls.cell_count / legendTotal) * 100).toFixed(1),
    }));

  // No Hazard = all cells not predicted as flooded (filtered-out + nodata + model code 0)
  const activeFloodedCount = hazardClasses
    .filter((cls) => cls.code >= 1)
    .reduce((sum, cls) => sum + cls.cell_count, 0);
  const noHazardPct = (((legendTotal - activeFloodedCount) / legendTotal) * 100).toFixed(1);
  const noHazardEntry = {
    code: 0, displayName: "No Hazard", depthRange: "No inundation",
    color: HAZARD_COLORS[0], pct: noHazardPct,
  };

  // Combined list: hazard classes first, No Hazard last
  const withPct = [...activePct, noHazardEntry];

  const filteredBarangays = barangayList.filter((n) =>
    n.toLowerCase().includes(barangaySearch.toLowerCase())
  );

  return (
    <div className="app-root">
      {loading && <LoadingOverlay />}

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

        <nav className="tab-nav">
          {Object.entries(TAB_LABELS).map(([tab, label]) => (
            <button key={tab} className={`tab-btn ${activeTab === tab ? "active" : ""}`}
              onClick={() => handleTabClick(tab)}>{label}</button>
          ))}
        </nav>

        <div className="tab-content">

          {/* ── INPUTS ──────────────────────────────────────────────────── */}
          {activeTab === "inputs" && (
            <div className="tab-pane">
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

              {barangayList.length === 0 && (
                <p className="empty-hint" style={{ marginTop: 8 }}>
                  Barangay boundaries unavailable. Check that the backend /api/barangays endpoint is running.
                </p>
              )}
            </div>
          )}

          {/* ── LEGEND ──────────────────────────────────────────────────── */}
          {activeTab === "legend" && (
            <div className="tab-pane">
              {!result ? (
                <p className="empty-hint">Run a prediction first to see the hazard legend.</p>
              ) : (
                <>
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

                  <div className="legend-list">
                    <h3 className="legend-title">Flood Hazard Classes</h3>
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

                  {highlightedName && (
                    <div className="brgy-selected-card">
                      <span className="brgy-selected-icon">📍</span>
                      <div>
                        <span className="brgy-selected-label">Selected Barangay</span>
                        <span className="brgy-selected-name">{highlightedName}</span>
                      </div>
                      <button className="brgy-selected-clear"
                        onClick={() => { setHighlightedName(""); setSelectedBarangay(null); }}>✕</button>
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* ── BARANGAY SUMMARY ────────────────────────────────────────── */}
          {activeTab === "summary" && (
            <div className="tab-pane">
              {!result ? (
                <p className="empty-hint">Run a prediction first to see the barangay hazard summary.</p>
              ) : summaryLoading ? (
                <div className="summary-loading">
                  <div className="btn-spinner" style={{ borderTopColor: "var(--cyan-400)" }} />
                  <p>Computing barangay summary…</p>
                </div>
              ) : barangaySummary.length === 0 ? (
                <p className="empty-hint">No barangay data available. Check that barangay boundaries are loaded.</p>
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
                          <th>Max Depth</th>
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
                              onClick={() => handleBarangayClick(b.name)}
                              style={{ cursor: "pointer" }}>
                              <td className="brgy-summary-name">{b.name}</td>
                              <td>
                                <span className="brgy-hazard-badge"
                                  style={{ background: HAZARD_COLORS[b.dominant] }}>
                                  {HAZARD_NAMES[b.dominant] ?? "—"}
                                </span>
                              </td>
                              <td className="brgy-summary-mono">
                                {b.maxDepth > 0 ? `${b.maxDepth.toFixed(2)}m` : "—"}
                              </td>
                              {/* NH (code 0) first, then Low–Extreme (codes 1–5) */}
                              {[0,1,2,3,4,5].map((c) => (
                                <td key={c} className="brgy-summary-count">
                                  {parseFloat(b.pcts?.[c] || 0) > 0
                                    ? <span style={{ color: HAZARD_COLORS[c], fontWeight: 600 }}>
                                        {b.pcts[c]}%
                                      </span>
                                    : <span style={{ opacity: 0.25 }}>—</span>
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
                    Values represent percent (%) of barangay area
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── HISTORY ─────────────────────────────────────────────────── */}
          {activeTab === "history" && (
            <div className="tab-pane">
              <div className="history-header">
                <h3>Past Scenarios</h3>
                <button className="refresh-btn" onClick={fetchHistory}>↻ Refresh</button>
              </div>
              {scenarios.length === 0
                ? <p className="empty-hint">No scenarios yet. Run a prediction to start building history.</p>
                : <div className="scenario-list">
                    {scenarios.map((s) => <ScenarioRow key={s.scenario_id} s={s} onLoad={loadScenario} />)}
                  </div>
              }
            </div>
          )}

          {/* ── DOWNLOADS / EXPORT ──────────────────────────────────────── */}
          {activeTab === "downloads" && (
            <div className="tab-pane">
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
            <div className="tab-pane">

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
                  ["Antecedent Ratio", "Antecedent ÷ Total Rain"],
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

        {!result && !barangayGeoJSON && (
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
            <LayersControl.BaseLayer name="Terrain">
              <TileLayer attribution="&copy; Google"
                url="https://{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}"
                subdomains={["mt0","mt1","mt2","mt3"]} maxZoom={20} />
            </LayersControl.BaseLayer>
            {overlayUrl && bounds && (
              <LayersControl.Overlay checked={overlayVisible} name="Flood Hazard Overlay">
                <ImageOverlay url={overlayUrl} bounds={bounds} opacity={overlayVisible ? opacity : 0} />
              </LayersControl.Overlay>
            )}
            {barangayGeoJSON && barangayVisible && (
              <LayersControl.Overlay checked name="Barangay Boundaries">
                <GeoJSON key={highlightedName} data={barangayGeoJSON}
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
        {result && hazardClasses.length > 0 && (
          <div className="map-legend-overlay">
            <div className="map-legend-title">Flood Hazard</div>
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
                      style={{ background: HAZARD_COLORS[bData.dominant] }}>
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
                  {bData.maxDepth > 0 && (
                    <div className="brgy-popup-depth">
                      Max depth: <strong>{bData.maxDepth.toFixed(2)} m</strong>
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
