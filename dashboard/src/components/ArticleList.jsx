// A filterable, paged list of one view (GET /live/articles?view=...).
import { useEffect, useMemo, useState } from "react";
import { api, errorText } from "../api";
import { NO_FILTERS } from "../filters";
import ArticleCard from "./ArticleCard";
import FilterBar from "./FilterBar";
import { Empty, Spinner } from "./ui";

const PAGE = 30;

function clean(params) {
  return Object.fromEntries(Object.entries(params).filter(([, v]) => v !== "" && v != null));
}

export default function ArticleList({ view, intro, narratives, sources, refreshKey, onChanged,
                                      reviewMode = false, emptyText = "Nothing here yet." }) {
  const [filters, setFilters] = useState(NO_FILTERS);
  const query = useMemo(() => clean({ view, ...filters }), [view, filters]);
  const qkey = JSON.stringify(query);
  const [page, setPage] = useState({ qkey: null, items: [], total: 0, error: null });
  const [more, setMore] = useState(false);

  // Re-fetch when the query changes, and on every refresh tick. A tick swaps the
  // data in place - no spinner - so a review form being filled in is not lost.
  useEffect(() => {
    let alive = true;
    api.articles({ ...query, limit: PAGE, offset: 0 })
      .then((d) => alive && setPage({ qkey, items: d.items, total: d.total, error: null }))
      .catch((e) => alive && setPage((p) => ({ ...p, qkey, error: errorText(e) })));
    return () => { alive = false; };
  }, [qkey, query, refreshKey]);

  const loadMore = async () => {
    setMore(true);
    try {
      const d = await api.articles({ ...query, limit: PAGE, offset: page.items.length });
      setPage((p) => ({ ...p, items: [...p.items, ...d.items], total: d.total }));
    } catch (e) {
      setPage((p) => ({ ...p, error: errorText(e) }));
    } finally {
      setMore(false);
    }
  };

  // A reviewed article may no longer belong to this view: the review queue only
  // holds unreviewed ones, and "not defence news" leaves every view but "dismissed".
  const reviewed = (updated) => {
    const leaves = view === "review"
      || (updated.category === "dismissed" && view !== "dismissed")
      || (view === "alerts" && updated.category !== "alert")
      || (view === "unconfirmed" && updated.category !== "unconfirmed");
    setPage((p) => ({
      ...p,
      total: leaves ? p.total - 1 : p.total,
      items: leaves ? p.items.filter((i) => i.id !== updated.id)
        : p.items.map((i) => (i.id === updated.id ? updated : i)),
    }));
    onChanged?.();
  };

  const loading = page.qkey !== qkey;

  return (
    <>
      {intro}
      <FilterBar filters={filters} setFilters={setFilters} narratives={narratives} sources={sources} />
      {page.error && <div className="banner error">{page.error}</div>}
      {loading ? <Spinner /> : (
        <>
          <div className="list-meta">
            <span>{page.items.length} of {page.total} shown</span>
          </div>
          {page.items.length === 0 ? <Empty>{emptyText}</Empty> : (
            <div className="cards">
              {page.items.map((item) => (
                <ArticleCard key={item.id} item={item} narratives={narratives}
                  startReviewing={reviewMode} onReviewed={reviewed} />
              ))}
            </div>
          )}
          {page.items.length < page.total && (
            <div style={{ textAlign: "center", marginTop: 18 }}>
              <button className="btn" disabled={more} onClick={loadMore}>
                {more ? "Loading…" : `Load ${Math.min(PAGE, page.total - page.items.length)} more`}
              </button>
            </div>
          )}
        </>
      )}
    </>
  );
}
