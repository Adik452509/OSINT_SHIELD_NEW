// Operations: start/stop the collection schedule, run one cycle now, and see
// what the last cycles did. Cycles run inside the API server (D24).
import { useCallback, useEffect, useState } from "react";
import { api, errorText } from "../api";
import { Empty, Panel } from "../components/ui";
import { C, localTime, timeAgo } from "../theme";

function CycleRow({ c }) {
  if (c.skipped) {
    return <tr><td colSpan={8} className="muted">skipped - {c.skipped}</td></tr>;
  }
  const t = c.ingest?.totals || c.totals || {};
  const f = c.fetch || {};
  const k = c.classify || {};
  const x = c.explain || {};
  const errs = c.errors || [];
  return (
    <tr className={errs.length ? "err" : ""}>
      <td title={localTime(c.started_at)}>{timeAgo(c.started_at)}</td>
      <td>{c.seconds != null ? `${c.seconds}s` : "—"}</td>
      <td>{t.new ?? "—"} / {t.relevant ?? "—"}</td>
      <td>{f.pages_ok ?? "—"}{f.pages_failed ? ` (${f.pages_failed} failed)` : ""}</td>
      <td>{f.gnews_decoded ?? "—"}{f.gnews_paused_until ? " · paused" : ""}</td>
      <td>{k.classified ?? "—"}</td>
      <td>{x.ok != null ? x.ok + (x.unverified || 0) : "—"}{x.seconds_each ? ` · ${x.seconds_each}s` : ""}</td>
      <td>{errs.length ? errs.join(" | ") : "—"}</td>
    </tr>
  );
}

export default function Operations({ onChanged }) {
  const [status, setStatus] = useState(null);
  const [every, setEvery] = useState(15);
  const [log, setLog] = useState([]);
  const [busy, setBusy] = useState(false);

  const addLog = (msg, kind = "info") =>
    setLog((l) => [...l.slice(-40), { msg, kind, time: new Date().toLocaleTimeString() }]);

  const refresh = useCallback(() => api.status().then(setStatus).catch(() => {}), []);

  useEffect(() => {
    let alive = true;
    const tick = () => api.status().then((s) => alive && setStatus(s)).catch(() => {});
    tick();
    const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const act = async (label, call) => {
    setBusy(true);
    addLog(`${label} …`);
    try {
      await call();
      addLog(`✓ ${label}`, "ok");
      onChanged?.();
    } catch (err) {
      addLog(`✕ ${label} - ${errorText(err)}`, "bad");
    } finally {
      setBusy(false);
      refresh();
    }
  };

  const s = status?.scheduler;
  const on = !!s?.scheduled;

  return (
    <div className="stack" style={{ gap: 16 }}>
      <div className="row">
        <Panel title="Collection schedule" dot={on ? C.low : C.high}>
          {!s ? <Empty>{status ? "This API server has no scheduler." : "loading…"}</Empty> : (
            <div className="stack">
              <div className="kv">
                <span>state</span>
                <span style={{ color: on ? C.low : C.high }}>{on ? `ON · every ${s.every_minutes} min` : "OFF"}</span>
              </div>
              <div className="kv"><span>cycle running now</span><span>{s.busy ? "yes" : "no"}</span></div>
              <div className="kv"><span>next cycle</span><span>{s.next_run_at ? localTime(s.next_run_at) : "—"}</span></div>
              <div className="kv"><span>last finished</span><span>{s.last_finished ? timeAgo(s.last_finished) : "—"}</span></div>
              {s.last_error && <div style={{ color: C.high, fontSize: 12 }}>last error: {s.last_error}</div>}
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginTop: 4 }}>
                <input type="number" min={1} max={1440} value={every} style={{ width: 70 }}
                  onChange={(e) => setEvery(Number(e.target.value) || 15)} disabled={on} />
                <span className="muted" style={{ fontSize: 12 }}>min</span>
                <button className="btn" disabled={busy || on} style={{ color: C.low }}
                  onClick={() => act("Start schedule", () => api.startScheduler(every))}>▶ Start</button>
                <button className="btn" disabled={busy || !on} style={{ color: C.high }}
                  onClick={() => act("Stop schedule", api.stopScheduler)}>■ Stop</button>
                <button className="btn" disabled={busy || s.busy} style={{ color: C.blue }}
                  onClick={() => act("Run one cycle now", api.runNow)}>⟳ Run now</button>
              </div>
              <div className="muted" style={{ fontSize: 11 }}>
                Stop lets a running cycle finish. Only one cycle runs at a time, even if
                scripts/11_live.py run is open in another window.
              </div>
            </div>
          )}
        </Panel>
        <Panel title="Pipeline" dot={C.amber}>
          <div className="stack">
            <div className="kv"><span>articles waiting for text</span><span>{status?.waiting_for_text ?? "—"}</span></div>
            <div className="kv">
              <span>Google News decoding</span>
              <span style={{ color: status?.google_news_paused_until && new Date(status.google_news_paused_until) > new Date() ? C.orange : C.low }}>
                {status?.google_news_paused_until && new Date(status.google_news_paused_until) > new Date()
                  ? `paused until ${localTime(status.google_news_paused_until)}` : "active"}
              </span>
            </div>
            <div className="muted" style={{ fontSize: 11, lineHeight: 1.6 }}>
              Each cycle: collect feeds → fetch article text → classify → Llama explains escalated
              articles with text (quotes checked word for word). Settings: configs/live.yaml.
            </div>
          </div>
        </Panel>
      </div>

      <Panel title="Recent cycles" dot={C.blue}>
        {!status?.recent_cycles?.length ? <Empty>No cycles yet.</Empty> : (
          <div style={{ overflowX: "auto" }}>
            <table className="table">
              <thead>
                <tr><th>started</th><th>took</th><th>new / relevant</th><th>pages fetched</th>
                  <th>Google decoded</th><th>classified</th><th>explained</th><th>errors</th></tr>
              </thead>
              <tbody>
                {status.recent_cycles.map((c, i) => <CycleRow key={c.started_at || i} c={c} />)}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <Panel title="Log" dot={C.amber}>
        <div className="log">
          {log.length === 0 && <div className="info">$ ready</div>}
          {log.map((l, i) => (
            <div key={i} className={l.kind}><span style={{ color: C.faint }}>[{l.time}]</span> {l.msg}</div>
          ))}
        </div>
      </Panel>
    </div>
  );
}
