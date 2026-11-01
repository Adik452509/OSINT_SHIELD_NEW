// Filters for an article list. Location filters from the old dashboard are gone:
// the pipeline does not extract countries or states, so they would filter on nothing.
import { useState } from "react";
import { NO_FILTERS } from "../filters";

export default function FilterBar({ filters, setFilters, narratives, sources, showSeverity = true }) {
  const [text, setText] = useState(filters.q);
  const set = (key, value) => setFilters((f) => ({ ...f, [key]: value }));
  const active = Object.entries(filters).some(([k, v]) => v && v !== NO_FILTERS[k]);

  return (
    <div className="filterbar">
      <span className="label">Filter</span>
      <select value={filters.narrative} onChange={(e) => set("narrative", e.target.value)}>
        <option value="">All narratives</option>
        {narratives.map((n) => <option key={n} value={n}>{n}</option>)}
      </select>
      {showSeverity && (
        <select value={filters.severity} onChange={(e) => set("severity", e.target.value)}>
          <option value="">All severities</option>
          <option value="High">High</option>
          <option value="Low">Low</option>
        </select>
      )}
      <select value={filters.source} onChange={(e) => set("source", e.target.value)}>
        <option value="">All sources</option>
        {sources.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
      <select value={filters.since_hours} onChange={(e) => set("since_hours", e.target.value)}>
        <option value="">Any time</option>
        <option value="6">Last 6 h</option>
        <option value="24">Last 24 h</option>
        <option value="168">Last 7 days</option>
      </select>
      <select value={filters.sort} onChange={(e) => set("sort", e.target.value)}>
        <option value="newest">Newest first</option>
        <option value="p_high">Highest P(High) first</option>
      </select>
      <input type="text" placeholder="search headlines… (Enter)" value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && set("q", text.trim())} />
      {active && (
        <button className="btn small" onClick={() => { setText(""); setFilters(NO_FILTERS); }}>
          ✕ Clear
        </button>
      )}
    </div>
  );
}
