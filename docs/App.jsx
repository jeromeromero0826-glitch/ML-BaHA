import { useState, useRef, useEffect, useCallback } from "react";
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  LayersControl,
  ScaleControl,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

// ─── Fit map to overlay bounds when result changes ───────────────────────────
function MapFitter({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) map.fitBounds(bounds, { padding: [24, 24] });
  }, [bounds, map]);
  return null;
}

// ─── Hazard colour swatch (driven by hazard_config colors) ───────────────────
function HazardSwatch({ color }) {
  return (
    <span
      className="legend-swatch"
      style={{ background: color || "#ccc" }}
    />
  );
}

// ─── Individual download button ───────────────────────────────────────────────
function DownloadBtn({ href, label, icon }) {
  if (!href) return null;
  return (
    <a className="dl-btn" href={href} download target="_blank" rel="noreferrer">
      <span className="dl-icon">{icon}</span>
      {label}
    </a>
  );
}

// ─── Scenario history row ─────────────────────────────────────────────────────
function ScenarioRow({ s, onLoad }) {
  const r = s.rainfall;
  return (
    <div className="scenario-row">
      <div className="scenario-meta">
        <span className="scenario-ts">{s.timestamp}</span>
        <span className="scenario-params">
          {r.depth}mm / {r.duration}h / API:{r.antecedent}mm
        </span>
      </div>
      <button className="scenario-load-btn" onClick={() => onLoad(s)}>
        Load
      </button>
    </div>
  );
}

// ─── Loading overlay ──────────────────────────────────────────────────────────
function LoadingOverlay() {
  return (
    <div className="loading-overlay" aria-live="polite">
      <div className="spinner-ring" />
      <p>Running XGBoost prediction…</p>
      <p className="loading-sub">Computing flood depth for all grid cells</p>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // Inputs
  const [duration,   setDuration]   = useState(12);
  const [depth,      setDepth]      = useState(180);
  const [antecedent, setAntecedent] = useState(60);

  // State
  const [result,    setResult]    = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");
  const [scenarios, setScenarios] = useState([]);

  // Map controls
  const [opacity,       setOpacity]       = useState(0.75);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [activeTab,     setActiveTab]     = useState("inputs"); // inputs | legend | history | downloads

  // Derived URLs
  const mapOutputs   = result?.map_outputs;
  const overlayUrl   = mapOutputs?.hazard_png ? `${API_BASE_URL}${mapOutputs.hazard_png}` : null;
  const bounds       = mapOutputs?.bounds || null;

  const outputUrls = {
    depthRaster:   result?.outputs?.depth_raster   ? `${API_BASE_URL}${result.outputs.depth_raster}`   : null,
    hazardRaster:  result?.outputs?.hazard_raster  ? `${API_BASE_URL}${result.outputs.hazard_raster}`  : null,
    predictionCsv: result?.outputs?.prediction_csv ? `${API_BASE_URL}${result.outputs.prediction_csv}` : null,
    summaryJson:   result?.outputs?.summary_json   ? `${API_BASE_URL}${result.outputs.summary_json}`   : null,
  };

  // Fetch scenario history
  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/scenarios?limit=20`);
      if (res.ok) {
        const data = await res.json();
        setScenarios(data.scenarios || []);
      }
    } catch (_) { /* history is optional */ }
  }, []);

  // Load a past scenario into state
  const loadScenario = (s) => {
    setResult({
      summary:            s.summary,
      hazard_class_counts:s.summary.hazard_class_counts,
      outputs:            s.outputs,
      map_outputs:        s.map_outputs,
    });
    setActiveTab("legend");
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          duration:   Number(duration),
          depth:      Number(depth),
          antecedent: Number(antecedent),
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Prediction request failed.");
      }

      const data = await res.json();
      setResult(data);
      setOverlayVisible(true);
      setActiveTab("legend");
      await fetchHistory();
    } catch (err) {
      setError(err.message || "Failed to fetch");
    } finally {
      setLoading(false);
    }
  };

  // Fetch history on mount
  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  const summary = result?.summary;
  const hazardClasses = result?.hazard_class_counts || [];

  return (
    <div className="app-root">
      {loading && <LoadingOverlay />}

      {/* ── SIDEBAR ──────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <span className="logo-icon">🌊</span>
            <div>
              <h1>FloodSight</h1>
              <p>XGBoost Hazard Prediction</p>
            </div>
          </div>
        </div>

        {/* Tab nav */}
        <nav className="tab-nav">
          {["inputs", "legend", "history", "downloads"].map((tab) => (
            <button
              key={tab}
              className={`tab-btn ${activeTab === tab ? "active" : ""}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab === "inputs"    && "⚙ Inputs"}
              {tab === "legend"    && "🎨 Legend"}
              {tab === "history"   && "🕘 History"}
              {tab === "downloads" && "⬇ Export"}
            </button>
          ))}
        </nav>

        <div className="tab-content">

          {/* ── INPUTS TAB ─────────────────────────────────────── */}
          {activeTab === "inputs" && (
            <div className="tab-pane">
              <form onSubmit={handlePredict} className="input-form">
                <div className="input-group">
                  <label htmlFor="duration">
                    Storm Duration
                    <span className="unit">hours</span>
                  </label>
                  <input
                    id="duration"
                    type="number"
                    value={duration}
                    onChange={(e) => setDuration(e.target.value)}
                    min="0" step="0.1" required
                  />
                </div>

                <div className="input-group">
                  <label htmlFor="depth">
                    Rainfall Depth
                    <span className="unit">mm</span>
                  </label>
                  <input
                    id="depth"
                    type="number"
                    value={depth}
                    onChange={(e) => setDepth(e.target.value)}
                    min="0" step="0.1" required
                  />
                </div>

                <div className="input-group">
                  <label htmlFor="antecedent">
                    Antecedent Rainfall
                    <span className="unit">mm</span>
                  </label>
                  <input
                    id="antecedent"
                    type="number"
                    value={antecedent}
                    onChange={(e) => setAntecedent(e.target.value)}
                    min="0" step="0.1" required
                  />
                </div>

                <button type="submit" className="predict-btn" disabled={loading}>
                  {loading ? (
                    <><span className="btn-spinner" /> Running…</>
                  ) : (
                    "▶ Run Prediction"
                  )}
                </button>
              </form>

              {error && <div className="error-box">{error}</div>}

              {summary && (
                <div className="stat-grid">
                  <div className="stat-card">
                    <span>Max Depth</span>
                    <strong>{summary.max_depth_m?.toFixed(3)} m</strong>
                  </div>
                  <div className="stat-card">
                    <span>Flooded Cells</span>
                    <strong>{summary.n_flooded_cells?.toLocaleString()}</strong>
                  </div>
                  <div className="stat-card">
                    <span>Mean Depth (flooded)</span>
                    <strong>{summary.mean_depth_flooded_cells_m?.toFixed(3)} m</strong>
                  </div>
                  <div className="stat-card">
                    <span>Total Cells</span>
                    <strong>{summary.n_cells?.toLocaleString()}</strong>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── LEGEND TAB ─────────────────────────────────────── */}
          {activeTab === "legend" && (
            <div className="tab-pane">
              {!result ? (
                <p className="empty-hint">Run a prediction first to see the hazard legend.</p>
              ) : (
                <>
                  <div className="map-controls">
                    <div className="control-row">
                      <label className="control-label">
                        <input
                          type="checkbox"
                          checked={overlayVisible}
                          onChange={(e) => setOverlayVisible(e.target.checked)}
                        />
                        Show hazard overlay
                      </label>
                    </div>

                    <div className="control-row">
                      <label className="control-label">
                        Opacity: <strong>{Math.round(opacity * 100)}%</strong>
                      </label>
                      <input
                        type="range"
                        min="0.1" max="1" step="0.05"
                        value={opacity}
                        onChange={(e) => setOpacity(Number(e.target.value))}
                        className="opacity-slider"
                        disabled={!overlayVisible}
                      />
                    </div>
                  </div>

                  <div className="legend-list">
                    <h3 className="legend-title">Flood Hazard Classes</h3>
                    {hazardClasses.map((cls) => (
                      <div className="legend-item" key={cls.code}>
                        <HazardSwatch color={cls.color} />
                        <div className="legend-text">
                          <span className="legend-name">{cls.name}</span>
                          <span className="legend-label">{cls.label}</span>
                        </div>
                        <span className="legend-count">
                          {cls.cell_count?.toLocaleString()}
                        </span>
                      </div>
                    ))}
                  </div>

                  <div className="scenario-id-box">
                    <span>Scenario ID</span>
                    <code>{result.outputs?.scenario_id}</code>
                  </div>
                </>
              )}
            </div>
          )}

          {/* ── HISTORY TAB ────────────────────────────────────── */}
          {activeTab === "history" && (
            <div className="tab-pane">
              <div className="history-header">
                <h3>Past Scenarios</h3>
                <button className="refresh-btn" onClick={fetchHistory}>↻ Refresh</button>
              </div>
              {scenarios.length === 0 ? (
                <p className="empty-hint">No scenarios yet. Run a prediction to start building history.</p>
              ) : (
                <div className="scenario-list">
                  {scenarios.map((s) => (
                    <ScenarioRow key={s.scenario_id} s={s} onLoad={loadScenario} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* ── DOWNLOADS TAB ──────────────────────────────────── */}
          {activeTab === "downloads" && (
            <div className="tab-pane">
              {!result ? (
                <p className="empty-hint">Run a prediction to enable downloads.</p>
              ) : (
                <div className="download-list">
                  <p className="dl-section-label">Raster outputs</p>
                  <DownloadBtn href={outputUrls.depthRaster}   label="Depth Raster (.tif)"   icon="📐" />
                  <DownloadBtn href={outputUrls.hazardRaster}  label="Hazard Raster (.tif)"  icon="🗺" />

                  <p className="dl-section-label">Tabular outputs</p>
                  <DownloadBtn href={outputUrls.predictionCsv} label="Prediction CSV"         icon="📊" />
                  <DownloadBtn href={outputUrls.summaryJson}   label="Summary JSON"           icon="📋" />
                </div>
              )}
            </div>
          )}

        </div>
      </aside>

      {/* ── MAP ──────────────────────────────────────────────────────────── */}
      <main className="map-area">
        {!result && (
          <div className="map-placeholder">
            <div className="placeholder-inner">
              <span className="placeholder-icon">🌊</span>
              <h2>No prediction yet</h2>
              <p>Enter rainfall parameters and click <strong>Run Prediction</strong> to generate the flood hazard map.</p>
            </div>
          </div>
        )}

        <MapContainer
          center={[14.5, 121.0]}
          zoom={10}
          scrollWheelZoom
          className="leaflet-map"
          zoomControl={true}
        >
          {bounds && <MapFitter bounds={bounds} />}

          <LayersControl position="topright">
            <LayersControl.BaseLayer checked name="OpenStreetMap">
              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
            </LayersControl.BaseLayer>

            <LayersControl.BaseLayer name="Satellite">
              <TileLayer
                attribution="&copy; Esri"
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                maxZoom={19}
              />
            </LayersControl.BaseLayer>

            <LayersControl.BaseLayer name="Terrain">
              <TileLayer
                attribution="&copy; Google"
                url="https://{s}.google.com/vt/lyrs=p&x={x}&y={y}&z={z}"
                subdomains={["mt0", "mt1", "mt2", "mt3"]}
                maxZoom={20}
              />
            </LayersControl.BaseLayer>

            {overlayUrl && bounds && (
              <LayersControl.Overlay
                checked={overlayVisible}
                name="Flood Hazard Overlay"
              >
                <ImageOverlay
                  url={overlayUrl}
                  bounds={bounds}
                  opacity={overlayVisible ? opacity : 0}
                />
              </LayersControl.Overlay>
            )}
          </LayersControl>

          <ScaleControl position="bottomleft" />
        </MapContainer>

        {/* Floating scenario stamp */}
        {result?.outputs?.scenario_id && (
          <div className="scenario-stamp">
            {result.outputs.scenario_id}
          </div>
        )}
      </main>
    </div>
  );
}
