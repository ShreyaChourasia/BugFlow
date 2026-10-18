// US-28
import { expect, test } from "vitest";

import { visibleWorkload, type WorkloadEntry } from "./workload";

const dev = (id: number, name: string, load: number, capacity: number): WorkloadEntry => ({
  developer_id: id,
  developer_name: name,
  current_queue_depth: load,
  capacity,
});

const entries = [dev(1, "Ada", 1, 5), dev(2, "Grace", 4, 5), dev(3, "Alan", 3, 3)];

test("shows the most loaded developers first, relative to capacity, capped", () => {
  expect(visibleWorkload(entries, "", 2).map((e) => e.developer_name)).toEqual(["Alan", "Grace"]);
});

test("a name search filters case-insensitively", () => {
  expect(visibleWorkload(entries, " a").map((e) => e.developer_name)).toEqual([
    "Alan",
    "Grace",
    "Ada",
  ]);
  expect(visibleWorkload(entries, "ADA").map((e) => e.developer_name)).toEqual(["Ada"]);
});
