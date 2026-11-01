// Overview: the numbers, the four categories of the alert policy, the latest
// alerts, and the 7-day trend.
import { useEffect, useState } from "react";
import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, errorText } from "../api";
import ArticleCard from "../components/ArticleCard";
import { Empty, MetricCard, Panel, SectionLabel } from "../components/ui";
import { C, NARRATIVE_COLOR, localTime, timeAgo } from "../theme";

const BOXES = [
  { view: "alerts", tab: "Alerts", label: "ALERTS", color: C.high,
    text: "High severity, full text available, not from an analysis outlet." },
  { view: "unconfirmed", tab: "Explorer", label: "UNCONFIRMED", color: C.orange,
    text: "High severity judged from the headline alone - the model is over-confident there." },
  { view: "analysis", tab: "Analysis", label: "ANALYSIS", color: C.blue,
    text: "Features and opinion (19FortyFive, The Diplomat). Often labelled High, rarely events." },
  { view: "review", tab: "Review", label: "TO REVIEW", color: C.amber,
    text: "Low-confidence or headline-only. Your corrections are how rare classes grow." },
];

function DarkTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border2}`, borderRadius: 8, padding: "8px 12px" }}>
      <div style={{ color: C.muted, fontSize: 11, marginBottom: 4 }}>{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color || C.text, fontSize: 13, fontWeight: 700 }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  );
}

export default function Overview({ summary, narratives, refreshKey, setTab }) {
  const [trend, setTrend] = useState([]);
  const [latest, setLatest] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    Promise.all([api.trend(7), api.articles({ view: "alerts", limit: 4 })])
      .then(([t, a]) => {
        if (!alive) return;
        setTrend(t.series);
        setLatest(a.items);
        setError(null);
      })
      .catch((e) => alive && setError(errorText(e)));
    return () => { alive = false; };
  }, [refreshKey]);

  const v = summary.views;
  const d = summary.last_24h;
  const narrData = Object.entries(summary.by_narrative).map(([name, count]) => ({ name, count }));
  const ex = summary.explanations || {};
  const cycle = summary.last_cycle;

  return (
    <>
      {error && <div className="banner error">{error}</div>}
      {summary.google_news_paused_until && new Date(summary.google_news_paused_until) > new Date() && (
        <div className="banner info">
          Google News decoding is paused until {localTime(summary.google_news_paused_until)} (Google
          rate-limited us). Those articles wait, then are classified from the headline alone.
        </div>
      )}

      <div className="metrics">
        <MetricCard label="Alerts · last 24 h" value={d.alerts} color={C.high}
          sub={`${v.alerts} in total`} onClick={() => setTab("Alerts")} />
        <MetricCard label="To review" value={v.review} color={C.amber}
          sub={`${v.dismissed} marked not defence news`} onClick={() => setTab("Review")} />
        <MetricCard label="Classified · last 24 h" value={d.all} color={C.blue}
          sub={`${summary.classified} in total · ${summary.waiting_for_text} waiting for text`} />
        <MetricCard label="LLM explanations" value={(ex.ok || 0) + (ex.unverified || 0)} color={C.low}
          sub={`${ex.ok || 0} with verified quotes`} />
      </div>

      <SectionLabel hint="click a box to open it">By category</SectionLabel>
      <div className="boxes">
        {BOXES.map((b) => (
          <div key={b.view} className="box" style={{ borderColor: `${b.color}44` }} onClick={() => setTab(b.tab)}>
            <div className="box-head" style={{ background: `${b.color}18`, borderBottom: `1px solid ${b.color}33` }}>
              <span className="box-name" style={{ color: b.color }}>{b.label}</span>
              <span className="box-count" style={{ background: `${b.color}33`, color: b.color }}>{v[b.view]}</span>
            </div>
            <div className="box-body">{b.text}</div>
          </div>
        ))}
      </div>

      <SectionLabel dot={C.high} hint={latest?.length ? "newest first" : undefined}>Latest alerts</SectionLabel>
      {latest === null ? <Empty>loading…</Empty>
        : latest.length === 0 ? <Empty>No alerts yet.</Empty>
        : <div className="cards">
            {latest.map((item) => <ArticleCard key={item.id} item={item} narratives={narratives} />)}
          </div>}

      <SectionLabel dot={C.low}>Activity</SectionLabel>
      <div className="row">
        <Panel title="Last 7 days (UTC)" dot={C.low} style={{ flex: 2 }}>
          {trend.length === 0 ? <Empty>No data yet.</Empty> : (
            <ResponsiveContainer width="100%" height={230}>
              <AreaChart data={trend}>
                <defs>
                  {[["gC", C.blue], ["gH", C.orange], ["gA", C.high]].map(([id, col]) => (
                    <linearGradient key={id} id={id} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={col} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={col} stopOpacity={0} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                <XAxis dataKey="day" tick={{ fill: C.faint, fontSize: 10 }} />
                <YAxis tick={{ fill: C.faint, fontSize: 10 }} allowDecimals={false} />
                <Tooltip content={<DarkTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Area type="monotone" dataKey="classified" name="Classified" stroke={C.blue} fill="url(#gC)" strokeWidth={2} />
                <Area type="monotone" dataKey="high" name="High" stroke={C.orange} fill="url(#gH)" strokeWidth={2} />
                <Area type="monotone" dataKey="alerts" name="Alerts" stroke={C.high} fill="url(#gA)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </Panel>
        <Panel title="Narratives" dot={C.accent}>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={narrData} layout="vertical" margin={{ left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border} horizontal={false} />
              <XAxis type="number" tick={{ fill: C.faint, fontSize: 10 }} allowDecimals={false} />
              <YAxis type="category" dataKey="name" tick={{ fill: C.muted, fontSize: 11 }} width={95} />
              <Tooltip content={<DarkTooltip />} cursor={{ fill: "#ffffff08" }} />
              <Bar dataKey="count" name="Articles" radius={[0, 5, 5, 0]}>
                {narrData.map((n) => <Cell key={n.name} fill={NARRATIVE_COLOR[n.name] || C.accent} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <div className="banner info" style={{ marginTop: 20 }}>
        <b>Reading the labels.</b> "High" severity follows the training labels, where it often means
        militarily significant rather than an incident happening now - hence the alert rules above.
        Propaganda is not reliably assessed. Model {summary.model_version}
        {cycle && <> · last cycle {timeAgo(cycle.finished_at)} ({cycle.seconds}s)</>}.
      </div>
    </>
  );
}
