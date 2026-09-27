# 001 — Store linked issue numbers on `Commit`, not a new `Issue` table

## Status
Accepted (Phase 2).

## Context
US-03 asks that fixing commits be matched to the issues they reference (via
patterns like `#123` or `fixes #123`) and that "the links" be stored. The
master prompt's §7 data model has no `Issue` entity — issues live in
whatever external tracker `Repository.issue_tracker_url` points at, and
BugFlow doesn't mirror them locally.

## Decision
Add `Commit.linked_issue_refs`: a nullable `ARRAY(Integer)` of the issue
numbers found in the commit message. No new table, no foreign key (there is
nothing local to point a foreign key at).

## Why
- The mining/labelling pipeline only needs to know *that* a commit
  references issue N, to (a) help decide it's a fix commit and (b) let a
  later phase cross-reference against a real tracker if one is connected.
  It doesn't need issue titles, state, or history in Phase 2.
- A full `Issue` entity mirroring an external tracker is exactly the kind of
  thing that becomes speculative infrastructure if nothing reads it yet —
  easy to add later (Phase 4 already talks to GitHub's REST API for PRs/checks;
  extending that integration to full issue sync is a natural, separate
  addition when a phase actually needs it).

## Consequences
- If a later phase wants real issue metadata (title, labels, state), it adds
  an `Issue` table then and can backfill from `linked_issue_refs`.
- `linked_issue_refs` is repository-relative issue numbers as they appear in
  commit messages — it does not distinguish two different repos' issue #7s.
  Not a problem today since it's only ever queried scoped to one `Commit`
  (which already has a `repository_id`).
