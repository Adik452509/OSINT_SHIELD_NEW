# OSINT Shield dashboard

React (Vite) front end for the live pipeline. Adapted from the original OSNIT dashboard: same
look, rewired to the real labels (M10d, see `docs/DECISIONS.md` D23–D25).

## Run

Two windows, both from the repo root:

```powershell
# 1. the API (and the collection scheduler)
.\.venv\Scripts\Activate.ps1
python scripts/10_serve.py --start

# 2. the dashboard
cd dashboard
npm install        # first time only
npm run dev        # → http://localhost:5173
```

The API address defaults to `http://127.0.0.1:8000`; set `VITE_API_URL` to change it.

## Tabs

| tab | shows |
|---|---|
| Overview | counts, the four categories, latest alerts, 7-day trend, narratives |
| Alerts | High severity, with full text, not from an analysis outlet (D23) |
| Analysis | 19FortyFive and The Diplomat — features and opinion |
| Review | unsure or headline-only articles: confirm, correct, or "Not defence news" |
| Explorer | every view, including unconfirmed (headline-only High) and dismissed |
| Operations | start / stop / run-now, recent cycles, Google News pause |

## Files

```
src/api.js                 every API call
src/theme.js               colours, categories, time formatting
src/App.jsx                header, tabs, shared data, 60 s refresh
src/tabs/Overview.jsx      metric cards, category boxes, charts
src/tabs/Operations.jsx    scheduler controls, cycle table, log
src/components/ArticleList.jsx   filterable, paged list of one view
src/components/ArticleCard.jsx   one article: labels, LLM summary + verified quotes, text
src/components/ReviewForm.jsx    confirm / correct / not defence news
src/components/FilterBar.jsx     narrative, severity, source, time, sort, search
```

The old dashboard's country/state filters, border-country panel and map are gone: the pipeline
does not extract locations, so they would have shown nothing real.
