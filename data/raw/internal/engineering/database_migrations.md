# Database migrations

Last reviewed: 2026-01-30 by Marcus Lee

## Tooling

All services use `golang-migrate` for schema migrations against Postgres 16 on RDS, except `billing-svc`, which uses Alembic since it is the one Python service with its own schema. Migration files live in `db/migrations/` inside each service directory in the monorepo, numbered sequentially.

## Writing a migration

Every migration needs a matching down migration, even if you never expect to run it. "We'll never roll this back" has been wrong before. Keep each migration to one logical change: adding a column, adding an index, backfilling a value. Do not combine a schema change with a large data backfill in the same migration; run the backfill as a separate, resumable job.

## Backward compatibility

Because deploys roll out gradually across pods, a migration must not break the currently running version of the service. This means:

1. Adding a column: make it nullable or give it a default. Do not add a NOT NULL column without a default in the same migration as code that requires it.
2. Removing a column: stop reading and writing it in application code first, deploy that, wait at least one full release cycle, then drop the column in a later migration.
3. Renaming a column: treat it as add-new, migrate reads and writes, backfill, then drop-old, across at least two separate deploys.
4. Changing a column type: usually needs a new column, a dual-write period, a backfill, then a cutover, same as a rename.

## Locking and large tables

`routes`, `drivers`, and `dispatch_events` are the largest tables, in the tens of millions of rows for larger fleet customers. Any migration touching these needs to avoid a full table lock. Use `CREATE INDEX CONCURRENTLY` for new indexes, never a plain `CREATE INDEX` in a migration against these tables. Adding a column with a non-null default on a large table in Postgres 16 is a fast metadata-only change as long as the default is constant, but double check this against the actual table before assuming it, since a volatile default forces a full rewrite.

## Review

Every migration PR needs a second reviewer from the data or platform team, regardless of how small the team authoring it is. This is a hard rule even for a one-line index addition, because a bad index build has locked a production table before.

## Running migrations

Migrations run automatically as an init container step before the new pods start serving traffic, as part of the normal Argo CD sync. If a migration fails, the deploy halts and the old pods keep serving; check the init container logs in Datadog or via `kubectl logs` on the failed pod.

## Staging first

Every migration must run successfully against staging before it can be merged for a prod release. The staging database is refreshed from a scrubbed prod snapshot weekly, so staging is a reasonable proxy for data volume, though not exact for the largest customers.

## Rollback

A failed migration on prod should be treated as a SEV2 at minimum if it blocks a deploy, higher if it left the schema in a partially applied state. Do not attempt to hand-fix schema state on prod without a second engineer confirming the plan first.
