// One classified article: labels, routing flags, the LLM's summary and verified
// quotes, the full text on demand, and the review form.
import { useState } from "react";
import { api, errorText } from "../api";
import { C, CATEGORY, NARRATIVE_COLOR, SEVERITY_COLOR, pct, timeAgo, localTime } from "../theme";
import ReviewForm from "./ReviewForm";
import { Badge } from "./ui";

function Explanation({ ex }) {
  if (!ex) return null;
  const verified = ex.status === "ok";
  return (
    <div className="llm">
      <div className="llm-label">
        LLM summary ({ex.model}) ·{" "}
        <span style={{ color: verified ? C.low : C.amber }}>
          {verified ? "quotes checked against the article" : "no quote could be verified"}
        </span>
      </div>
      {ex.summary && <p>{ex.summary}</p>}
      {ex.evidence.map((q, i) => <p key={i} className="quote">“{q}”</p>)}
    </div>
  );
}

function ReviewNote({ item }) {
  const r = item.review;
  if (!r) return null;
  if (r.not_relevant) return <div className="reviewed">✕ You marked this: not defence news</div>;
  const changes = [];
  if (item.effective.narrative !== item.narrative.label)
    changes.push(`narrative ${item.narrative.label} → ${item.effective.narrative}`);
  if (item.effective.severity !== item.severity.label)
    changes.push(`severity ${item.severity.label} → ${item.effective.severity}`);
  return (
    <div className="reviewed">
      ✓ Reviewed {timeAgo(r.reviewed_at)}: {changes.length ? changes.join(", ") : "model confirmed"}
      {r.note && <span className="muted"> — {r.note}</span>}
    </div>
  );
}

export default function ArticleCard({ item, narratives, startReviewing = false, onReviewed }) {
  const [reviewing, setReviewing] = useState(startReviewing);
  const [detail, setDetail] = useState(null);
  const [showText, setShowText] = useState(false);
  const [error, setError] = useState(null);

  const cat = CATEGORY[item.category];
  const sev = item.effective.severity;
  const sevColor = SEVERITY_COLOR[sev];
  const narr = item.effective.narrative;
  const conf = item.narrative.source === "rubric" ? "keyword rule"
    : item.narrative.confidence != null ? pct(item.narrative.confidence) : "";

  const toggleText = async () => {
    if (showText) return setShowText(false);
    setShowText(true);
    if (!detail) {
      try {
        setDetail(await api.article(item.id));
      } catch (err) {
        setError(errorText(err));
      }
    }
  };

  return (
    <div className="card" style={{ borderColor: cat ? `${cat.color}55` : undefined }}>
      <div className="stripe" style={{ background: `linear-gradient(90deg, ${sevColor}, ${sevColor}33)` }} />
      <div className="card-body">
        <div className="tags">
          {cat && <Badge color={cat.color}>{cat.label}</Badge>}
          <Badge color={sevColor}
            title="Severity as the training labels define it - often 'militarily significant', not only 'incident now' (D23)">
            {sev}
          </Badge>
          <span className="chip" style={{ color: NARRATIVE_COLOR[narr] }}>
            {narr} {conf && <span className="muted">· {conf}</span>}
          </span>
          {!item.has_body && <Badge color={C.dim} title="Only the headline was available">headline only</Badge>}
          {item.needs_review && !item.review && <Badge color={C.amber}>needs review</Badge>}
          <span className="phigh" style={{ color: sevColor, background: `${sevColor}18` }}
            title="Model probability of High severity">
            P(High) {pct(item.severity.p_high)}
          </span>
        </div>

        <p className="headline">{item.headline}</p>
        <div className="meta">
          <span>{item.source_domain}</span>
          <span title={localTime(item.published_at || item.collected_at)}>
            {timeAgo(item.published_at || item.collected_at)}
          </span>
          {item.matched_terms.length > 0 && <span>matched: {item.matched_terms.join(", ")}</span>}
        </div>

        <Explanation ex={item.explanation} />
        <ReviewNote item={item} />

        {showText && (
          <div className="body-text">
            {error ? <span style={{ color: C.high }}>{error}</span>
              : !detail ? "loading…"
              : detail.body || detail.feed_text
              || `No article text (${detail.fetch_error || "headline only"}).`}
          </div>
        )}

        {reviewing && (
          <ReviewForm item={item} narratives={narratives}
            onCancel={() => setReviewing(false)}
            onDone={(updated) => { setReviewing(false); onReviewed?.(updated); }} />
        )}

        <div className="card-actions">
          <button className="btn link" onClick={toggleText}>
            {showText ? "▲ Hide text" : item.has_body ? "▼ Full text" : "▼ Details"}
          </button>
          {!reviewing && (
            <button className="btn link" style={{ color: C.amber }} onClick={() => setReviewing(true)}>
              ✎ Review
            </button>
          )}
          <span className="spacer" />
          <a href={item.url} target="_blank" rel="noreferrer">Source →</a>
        </div>
      </div>
    </div>
  );
}
