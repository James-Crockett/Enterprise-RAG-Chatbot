# Observability

Last reviewed: 2026-02-18 by Marcus Lee

## Stack

Datadog is the primary tool for metrics, logs, and traces across all services. PagerDuty handles alert routing and paging, integrated with Datadog monitors and with the incident channels described in `incident_management.md`. Every service in `kestrel/platform` ships structured JSON logs and APM traces by default through the shared Datadog agent sidecar; there is no service-specific opt-out without a documented exception from Marcus Lee.

## Dashboards

Each service has a standard dashboard named `<service>-oncall`, created from a shared template so on-call engineers do not need to relearn a layout for each service. The template includes request rate, error rate, p50 and p99 latency, and, for services with a database, connection pool usage. Team-specific dashboards can exist alongside this, but the `-oncall` one should never be deleted or renamed since PagerDuty runbook links point to it directly.

## Alerting

Alerts fire off Datadog monitors, not raw metrics thresholds set ad hoc in code. Every alert needs an owner and a linked runbook section; an alert with no owner gets muted automatically after 30 days by a cleanup job Sam Ortiz's team runs. Noisy alerts should be tuned or deleted within a week of someone flagging them in `#eng-observability`, since alert fatigue is the fastest way to miss a real SEV1.

Standard thresholds: page on 5xx rate above 1 percent over 5 minutes for public-facing services, page on p99 latency above 2 seconds sustained for 10 minutes, page on Kafka consumer lag above 10,000 messages for `dispatch-worker`.

## Tracing

APM traces use Datadog's Go and Python tracers, auto-instrumented for HTTP and database calls. Custom spans should be added around any external call, like the calls `dispatch-worker` makes to `geo-tiles`, so a slowdown shows up clearly attributed rather than as generic latency. Trace sampling is 100 percent in staging and dev, 10 percent in prod except for traces tied to an error, which are always kept.

## Logs

Log level in prod is `info` by default, `debug` only temporarily and only scoped to a specific service or even a specific pod through a LaunchDarkly-gated flag, since `debug` logging at fleet scale gets expensive fast in Datadog's ingestion pricing. Never log full request bodies for `billing-svc` or anything touching payment details; log a reference ID and look up the rest from the database if needed.

## SLOs

`route-api` has a published SLO of 99.9 percent availability measured monthly, tracked on its own Datadog SLO dashboard. `dispatch-worker` is tracked on job completion latency rather than availability, since it is not directly request-driven. SLO burn alerts page the same rotation as the underlying service's other alerts, not a separate one.

## Onboarding

New engineers get Datadog access provisioned automatically through Okta SSO on their first day, as part of the standard access group for their team. If a dashboard or monitor is missing after onboarding, file a ticket through the IT portal rather than asking around in Slack, so Sam Ortiz's team has a record of the gap.
