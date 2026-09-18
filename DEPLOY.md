# Deploying ML-BaHa

Two services, and **the order matters**.

- **Backend** — FastAPI on Render, at `https://ml-baha.onrender.com`
- **Frontend** — React/Vite on Vercel, at `https://ml-ba-ha.vercel.app`

> **Deploy the backend first, then the frontend.**
> The frontend reads `barangay_summary` and `summary.area_totals` out of the
> `/api/predict` response. If Vercel updates while Render is still running older
> code, the legend renders empty and the Summary tab reports no data. In the other
> order the gap is harmless: the old frontend simply ignores the new fields.

---

## 1. Before you commit

Your working tree contains changes from more than one sitting, so read the diff
before staging anything.

```bash
git status
git diff
```

Files that must be committed or the backend will not start:

```
backend/assets/tables/barangay_meta.json
backend/assets/tables/barangay_index.npz
backend/app/services/barangay_summary.py
```

`asset_loader` fails fast at startup when the index is missing, with a message
telling you to run the generator. That is deliberate: the alternative is a server
that boots happily and then errors on the first prediction.

Also new:

```
backend/scripts/build_barangay_index.py
backend/requirements-dev.txt
frontend/src/ErrorBoundary.jsx
```

`backend/outputs/` is now gitignored. It holds generated rasters, PNGs and CSVs
that are cleared at startup and rotated per scenario.

```bash
git add -A
git commit -m "Server-side barangay summary, area-based percentages, reliability fixes"
```

You are on branch `claude/setup-ml-baha-webapp-777eO`, whose upstream is gone.
Merge into `main` before deploying if that is what Render and Vercel build from:

```bash
git checkout main
git merge claude/setup-ml-baha-webapp-777eO
git push origin main
```

## 2. Regenerating the barangay index

Only needed when the barangay boundaries or the spatial grid change. The output
is committed, so a normal deploy does not run it.

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python scripts/build_barangay_index.py
```

Expected output:

```
37 barangays
479,272 cells in the model grid extent
193,718 cells fall inside Sipocot (40.4% of the grid extent)
68,276 active cells, 68,227 inside Sipocot
7,490 active cells flagged as river channel (log10 flow accumulation >= 4.0)
```

If those numbers move, something upstream changed and the legend percentages
will shift with them.

## 3. Backend to Render

Push, or trigger a manual deploy. Watch the logs for the startup banner:

```
[asset_loader] Assets loaded: 68,276 grid cells | 12 features | 6 hazard classes
               | 37 barangays (193,718 cells inside Sipocot)
Assets loaded successfully.
```

If instead you see `Barangay index not found`, the two asset files did not make
it into the commit.

Then check it directly:

```bash
curl https://ml-baha.onrender.com/api/health
# {"status":"ok"}

curl -X POST https://ml-baha.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"duration":20,"depth":550,"antecedent":50}' | head -c 400
```

The response must contain `barangay_summary` and `summary.area_totals`. The first
call after an idle period takes 20 to 30 seconds while the instance wakes.

## 4. Frontend to Vercel

`frontend/.env.production` points at the Render URL, but **Vercel's own
environment variable overrides the file**. Check them both, or they drift:

Vercel project → Settings → Environment Variables → `VITE_API_BASE_URL`
should be `https://ml-baha.onrender.com`.

Build locally once before pushing, since CI will not catch a syntax error for you:

```bash
cd frontend
npm install     # html2canvas and jspdf were removed; this refreshes package-lock.json
npm run build
```

Then push, or trigger the deploy.

## 5. Live smoke test

Work through this on the deployed site, not locally.

- [ ] Page loads
- [ ] Browser tab reads "ML-BaHa — Flood Hazard Prediction for Sipocot, Camarines Sur"
- [ ] Before the first prediction, the "No prediction yet" card sits over the map
      as a card, with the barangay boundaries still visible around it
- [ ] Barangay boundaries appear on the map without a reload
- [ ] **Run Prediction** with the defaults (180 mm / 12 h / 60 mm) returns a map
- [ ] On a cold start the overlay says the server is waking, not just "Running"
- [ ] Legend shows six classes as percentages of **Sipocot, 174.3 km²**
- [ ] Legend percentages sum to 100
- [ ] **Summary** tab fills immediately, 37 barangays, no spinner and no error
- [ ] Clicking a barangay switches both the sidebar legend and the map legend box
      to that barangay, with its area in the header
- [ ] **Back to all of Sipocot** restores the municipal view
- [ ] Depth note under the legend shows max on land, and the in-channel figure
      where the barangay has river cells
- [ ] Base layers: OpenStreetMap, Satellite, **Terrain from Esri** (not Google)
- [ ] Run a second prediction with different rainfall, then load the first one
      from **History**: its map and downloads must still work
- [ ] All four downloads in **Export** open
- [ ] Check it on a phone as well as the laptop

## 6. If something fails on stage

- Backend asleep and the demo is starting: open the site a few minutes early, or
  hit `/api/health` from your phone, to wake it.
- Backend unreachable: have screenshots of the map, the Summary tab and a
  barangay breakdown ready in the slides.
- A render error now shows a card with a reload button instead of a blank page.
  Reloading genuinely does usually clear it.

## Known limitations

- The free Render instance sleeps when idle. The first request after a pause
  takes 20 to 30 seconds.
- Only the five most recent scenarios keep their files. Older ones drop out of
  History rather than lingering as dead links.
- `No Hazard` sits near 65% of municipal area for any rainfall, because roughly
  two thirds of Sipocot's cells were filtered out as permanently dry during
  training and are counted as No Hazard by definition.
