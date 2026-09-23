# On-call runbook

Last reviewed: 2026-03-02 by Marcus Lee

## Rotation

On-call is weekly, handoff Monday 10:00 Central in the `#oncall-handoff` Slack channel. The outgoing engineer posts a summary of open issues, anything flapping, and any deploys still being watched. Stipend is $400 per on-call week, paid through the next payroll cycle after the week ends; check Workday if it does not show up within two cycles.

There are separate rotations per service: `route-api-oncall`, `dispatch-worker-oncall`, `billing-svc-oncall`, and a combined `frontend-oncall` covering `web-app` and `mobile-driver`. `geo-tiles` is covered by `dispatch-worker-oncall` since the two teams overlap.

## Getting paged

PagerDuty pages go to your phone and to Slack. Acknowledge within the SLA for the severity (see `incident_management.md` for SEV definitions): 15 minutes for SEV1, 30 minutes for SEV2. If you do not ack, PagerDuty escalates to the secondary after 10 minutes, then to the EM after another 10.

## First 10 minutes

1. Acknowledge the page.
2. Open the Datadog dashboard for the affected service. Each service has a saved dashboard named `<service>-oncall`.
3. Check for a recent deploy. Argo CD history is the fastest way; if a deploy went out in the last hour, rolling it back is usually the right first move rather than debugging forward.
4. If customer-facing, open an incident channel named `#inc-YYYYMMDD-short-name` and post the initial summary there.
5. Decide severity and follow the paging and comms steps in `incident_management.md`.

## Common issues by service

Route-api: elevated 5xx usually traces back to a Postgres connection pool exhaustion. Check `pg_stat_activity` count against the RDS max connections limit before assuming it is application code.

Dispatch-worker: consumer lag on `dispatch.jobs` climbing usually means a downstream call to `geo-tiles` is slow. Check `geo-tiles` p99 latency in Datadog first.

Billing-svc: failed usage aggregation jobs are almost always a Kafka offset issue after a redeploy. Rerunning the aggregation job for the affected window from the admin CLI (`billing-svc admin reprocess --window <start>..<end>`) is the standard fix.

Web-app: a spike in client errors right after a deploy is almost always a stale CDN cache serving an old bundle referencing a removed asset. Purge the CloudFront distribution for `web-app` and confirm.

## Escalation contacts

Marcus Lee for anything cross-service or ambiguous. Tom Brennan for anything that looks like a security incident, including suspicious auth activity in Okta logs. Priya Raman if a decision needs to override the normal release window or freeze rules.

## Access

On-call engineers get standing read access to prod Datadog and logs. Anything requiring a prod shell or database access still needs a ServiceNow access portal request, though during an active SEV1 or SEV2 Tom Brennan's team can grant emergency break-glass access, logged and reviewed the next business day.

## Handoff checklist

Before handing off, make sure any open incident has a named owner who is not you, PagerDuty schedule overrides are cleared if you added any, and the on-call notes doc in Confluence is updated with anything the next person should know.
