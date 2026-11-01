// OSINT Shield dashboard: header, tabs, and the data every tab shares.
import { useEffect, useState } from "react";
import { API_URL, api, errorText } from "./api";
import ArticleList from "./components/ArticleList";
import { Spinner } from "./components/ui";
import Operations from "./tabs/Operations";
import Overview from "./tabs/Overview";
import { C, timeAgo } from "./theme";

const TABS = ["Overview", "Alerts", "Analysis", "Review", "Explorer", "Operations"];
const TAB_VIEW = { Alerts: "alerts", Analysis: "analysis", Review: "review" };
const REFRESH_MS = 60_000;

const EXPLORER_VIEWS = [
  ["all", "Everything"],
  ["alerts", "Alerts"],
  ["unconfirmed", "Unconfirmed (headline-only High)"],
  ["analysis", "Analysis outlets"],
  ["review", "Review queue"],
  ["dismissed", "Not defence news"],
];

function Intro({ children }) {
  return <div className="banner info">{children}</div>;
}

function Explorer(props) {
  const [view, setView] = useState("all");
  return (
    <ArticleList key={view} view={view} {...props}
      intro={
        <div className="filterbar">
          <span className="label">View</span>
          <select value={view} onChange={(e) => setView(e.target.value)}>
            {EXPLORER_VIEWS.map(([v, label]) => <option key={v} value={v}>{label}</option>)}
          </select>
        </div>
      } />
  );
}

export default function App() {
  const [tab, setTab] = useState("Overview");
  const [summary, setSummary] = useState(null);
  const [scheduler, setScheduler] = useState(null);
  const [meta, setMeta] = useState({ narratives: [], sources: [] });
  const [error, setError] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [loadedAt, setLoadedAt] = useState(null);

  useEffect(() => {
    let alive = true;
    Promise.all([api.summary(), api.sources(), api.status()])
      .then(([s, src, st]) => {
        if (!alive) return;
        setSummary(s);
        setMeta({ narratives: src.narratives,
                  sources: src.sources.filter((x) => x.classified > 0).map((x) => x.source) });
        setScheduler(st.scheduler);
        setError(null);
        setLoadedAt(new Date());
      })
      .catch((e) => alive && setError(errorText(e)));
    return () => { alive = false; };
  }, [refreshKey]);

  useEffect(() => {
    const id = setInterval(() => setRefreshKey((k) => k + 1), REFRESH_MS);
    return () => clearInterval(id);
  }, []);

  const refresh = () => setRefreshKey((k) => k + 1);
  const on = !!scheduler?.scheduled;
  const listProps = { narratives: meta.narratives, sources: meta.sources, refreshKey, onChanged: refresh };

  return (
    <div>
      <header className="header">
        <div className="logo">
          <img src="/shield.svg" alt="" width={22} height={22} />
          <span>OSINT <strong>Shield</strong></span>
        </div>
        <div className="pill">
          <span className="dot" style={{ background: on ? C.low : C.high, boxShadow: on ? `0 0 6px ${C.low}` : "none" }} />
          collection: <strong style={{ color: on ? C.low : C.high }}>{on ? "ON" : "OFF"}</strong>
          {scheduler?.busy && <span style={{ color: C.amber }}>· running</span>}
        </div>
        <button className="btn small" style={{ color: C.accent }} onClick={refresh}>↺ Refresh</button>
        <div className="header-right">
          {summary && <>{summary.classified} classified · </>}
          updated {loadedAt ? timeAgo(loadedAt.toISOString()) : "—"}
        </div>
      </header>

      <nav className="nav">
        {TABS.map((t) => (
          <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
            {t}
            {summary && TAB_VIEW[t] && <span className="count">{summary.views[TAB_VIEW[t]]}</span>}
          </button>
        ))}
      </nav>

      <main className="main">
        {error && (
          <div className="banner error">
            {error}. Is the API running? Start it with <code>python scripts/10_serve.py</code> ({API_URL}).
          </div>
        )}
        {!summary ? (!error && <Spinner label="LOADING OSINT SHIELD…" />) : (
          <>
            {tab === "Overview" && (
              <Overview summary={summary} narratives={meta.narratives} refreshKey={refreshKey} setTab={setTab} />
            )}
            {tab === "Alerts" && (
              <ArticleList view="alerts" {...listProps} emptyText="No alerts."
                intro={<Intro>High severity, with full article text, from news outlets (analysis
                  outlets are under Analysis). Severity follows the training labels - review anything
                  that looks wrong.</Intro>} />
            )}
            {tab === "Analysis" && (
              <ArticleList view="analysis" {...listProps}
                intro={<Intro>Features and opinion from 19FortyFive and The Diplomat. The training
                  labels call many of them High, so they are kept apart from alerts.</Intro>} />
            )}
            {tab === "Review" && (
              <ArticleList view="review" {...listProps} reviewMode emptyText="The review queue is empty."
                intro={<Intro>The model was unsure, or saw only a headline. Confirm or correct each
                  label, or mark it <b>Not defence news</b>. Your answers are stored alongside the
                  model's - never over it - and become training data.</Intro>} />
            )}
            {tab === "Explorer" && <Explorer {...listProps} />}
            {tab === "Operations" && <Operations onChanged={refresh} />}
          </>
        )}
      </main>
    </div>
  );
}
