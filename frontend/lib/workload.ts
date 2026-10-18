export type WorkloadEntry = {
  developer_id: number;
  developer_name: string;
  capacity: number;
  current_queue_depth: number;
};

export const DEFAULT_VISIBLE_DEVELOPERS = 20;

const loadRatio = (e: WorkloadEntry) =>
  e.capacity > 0 ? e.current_queue_depth / e.capacity : e.current_queue_depth;

// US-28: hundreds of mined developers make one chart unreadable — show the
// most-loaded first (relative to capacity), optionally narrowed by a name
// search, capped either way.
export function visibleWorkload(
  entries: WorkloadEntry[],
  query: string,
  limit = DEFAULT_VISIBLE_DEVELOPERS,
): WorkloadEntry[] {
  const q = query.trim().toLowerCase();
  const sorted = [...entries].sort((a, b) => loadRatio(b) - loadRatio(a));
  return q
    ? sorted.filter((e) => e.developer_name.toLowerCase().includes(q)).slice(0, limit)
    : sorted.slice(0, limit);
}
