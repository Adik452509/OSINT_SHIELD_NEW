// An analyst's verdict on one article (POST /live/articles/{id}/review).
//
// "Save" stores the labels as shown - change them first to correct the model, or
// save unchanged to confirm it was right (that also clears it from the review queue).
// "Not defence news" removes the article from every view but "dismissed"; those
// verdicts are the labelled examples a better relevance filter can learn from (D22).
// The model's own prediction is never overwritten (D24).
import { useState } from "react";
import { api, errorText } from "../api";
import { C, NARRATIVE_COLOR, SEVERITY_COLOR } from "../theme";

export default function ReviewForm({ item, narratives, onDone, onCancel }) {
  const [narrative, setNarrative] = useState(item.effective.narrative);
  const [severity, setSeverity] = useState(item.effective.severity);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const send = async (body) => {
    setBusy(true);
    setError(null);
    try {
      onDone(await api.review(item.id, { ...body, note: note.trim() || null }));
    } catch (err) {
      setError(errorText(err));
      setBusy(false);
    }
  };

  const changed = narrative !== item.narrative.label || severity !== item.severity.label;

  return (
    <div className="review-form">
      <div className="fields">
        <select value={narrative} onChange={(e) => setNarrative(e.target.value)}
          style={{ color: NARRATIVE_COLOR[narrative] }}>
          {narratives.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <div className="seg">
          {["High", "Low"].map((s) => (
            <button key={s} type="button" className={severity === s ? "on" : ""}
              style={severity === s ? { background: SEVERITY_COLOR[s] } : undefined}
              onClick={() => setSeverity(s)}>{s}</button>
          ))}
        </div>
        <input type="text" placeholder="note (optional)" value={note} maxLength={1000}
          onChange={(e) => setNote(e.target.value)} />
      </div>
      <div className="buttons">
        <button className="btn small" disabled={busy}
          style={{ color: C.low, borderColor: `${C.low}66` }}
          onClick={() => send({ narrative, severity })}>
          {changed ? "✓ Save correction" : "✓ Model is right"}
        </button>
        <button className="btn small" disabled={busy}
          style={{ color: C.high, borderColor: `${C.high}66` }}
          onClick={() => send({ not_relevant: true })}>
          ✕ Not defence news
        </button>
        <button className="btn small" disabled={busy} onClick={onCancel}>Cancel</button>
      </div>
      {error && <div style={{ color: C.high, fontSize: 12, marginTop: 8 }}>{error}</div>}
    </div>
  );
}
