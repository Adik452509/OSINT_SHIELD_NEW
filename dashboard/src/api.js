// Every call the dashboard makes to the OSINT Shield API (scripts/10_serve.py).
// Endpoints are documented at http://127.0.0.1:8000/docs and in src/osint_shield/api/live.py.
import axios from "axios";

export const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const http = axios.create({ baseURL: API_URL, timeout: 20000 });
const data = (r) => r.data;
const enc = encodeURIComponent;

export const api = {
  summary: () => http.get("/live/summary").then(data),
  // params: view, narrative, severity, source, q, since_hours, sort, limit, offset
  articles: (params) => http.get("/live/articles", { params }).then(data),
  article: (id) => http.get(`/live/articles/${enc(id)}`).then(data),
  // body: {narrative, severity, note} or {not_relevant: true, note}
  review: (id, body) => http.post(`/live/articles/${enc(id)}/review`, body).then(data),
  trend: (days = 7) => http.get("/live/trend", { params: { days } }).then(data),
  sources: () => http.get("/live/sources").then(data),
  status: () => http.get("/live/status").then(data),
  startScheduler: (everyMinutes) =>
    http.post("/live/scheduler/start", everyMinutes ? { every_minutes: everyMinutes } : {}).then(data),
  stopScheduler: () => http.post("/live/scheduler/stop").then(data),
  runNow: () => http.post("/live/run-now").then(data),
};

/** A readable message from a failed request. */
export function errorText(err) {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; ");
  if (err?.code === "ERR_NETWORK") return `cannot reach the API at ${API_URL}`;
  return err?.message || "request failed";
}
