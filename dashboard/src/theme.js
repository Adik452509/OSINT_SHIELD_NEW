// Colours and small formatting helpers shared by every component.
// The palette is the old OSNIT dashboard's (GitHub-dark), kept on purpose.

export const C = {
  bg: "#010409",
  panel: "#0d1117",
  panel2: "#161b22",
  border: "#21262d",
  border2: "#30363d",
  text: "#e6edf3",
  text2: "#c9d1d9",
  muted: "#8b949e",
  dim: "#6e7681",
  faint: "#484f58",
  accent: "#6366f1",
  high: "#ff2d55",
  low: "#30d158",
  amber: "#ffd60a",
  orange: "#ff6b35",
  blue: "#0a84ff",
};

// The alert policy's categories (D23): what each article is on the dashboard.
export const CATEGORY = {
  alert: { label: "ALERT", color: C.high },
  unconfirmed: { label: "UNCONFIRMED", color: C.orange },
  analysis: { label: "ANALYSIS", color: C.blue },
  dismissed: { label: "NOT DEFENCE", color: C.faint },
  normal: null,
};

export const NARRATIVE_COLOR = {
  Security: "#6366f1",
  Political: "#ec4899",
  Civilian: "#14b8a6",
  Investigation: "#f97316",
  Other: "#8b949e",
};

export const SEVERITY_COLOR = { High: C.high, Low: C.low };

/** "5 min ago", "3 h ago", "2 d ago" from an ISO timestamp (UTC, as the API stores them). */
export function timeAgo(iso) {
  if (!iso) return "";
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return `${Math.floor(seconds / 86400)} d ago`;
}

/** Local date and time for an ISO timestamp. */
export function localTime(iso) {
  return iso ? new Date(iso).toLocaleString() : "—";
}

export const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
