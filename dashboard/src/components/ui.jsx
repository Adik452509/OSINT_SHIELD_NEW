// Small building blocks, kept from the old OSNIT dashboard.
import { C } from "../theme";

export function Panel({ title, dot = C.accent, children, style }) {
  return (
    <div className="panel" style={style}>
      {title && (
        <div className="panel-title">
          <span className="dot" style={{ background: dot }} />
          {title}
        </div>
      )}
      {children}
    </div>
  );
}

export function SectionLabel({ children, hint, dot = C.accent }) {
  return (
    <div className="section-label">
      <span className="dot" style={{ background: dot }} />
      {children}
      {hint && <span className="hint">— {hint}</span>}
    </div>
  );
}

export function MetricCard({ label, value, color, sub, onClick }) {
  return (
    <div className="metric" style={{ borderColor: `${color}44`, cursor: onClick ? "pointer" : "default" }}
      onClick={onClick}>
      <div className="corner" style={{ background: color }} />
      <div className="label">{label}</div>
      <div className="value" style={{ color }}>{value ?? 0}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Badge({ children, color, title }) {
  return (
    <span className="badge" title={title}
      style={{ background: `${color}22`, color, borderColor: `${color}44` }}>
      {children}
    </span>
  );
}

export function Spinner({ label }) {
  return (
    <div className="center">
      <div className="spinner" />
      {label}
    </div>
  );
}

export function Empty({ children }) {
  return <div className="empty">{children}</div>;
}
