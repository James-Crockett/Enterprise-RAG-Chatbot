# Feature flags

Last reviewed: 2026-02-12 by Nina Castellanos

## Tooling

Kestrel Route uses LaunchDarkly for all feature flags across `route-api`, `dispatch-worker`, `billing-svc`, and `web-app`. The mobile driver app reads flag state through a cached snapshot fetched on app start, refreshed every 15 minutes, since it cannot always maintain a live streaming connection in the field.

## Naming

Flags are named `<service>-<short-description>`, all lowercase, hyphenated, for example `route-api-multi-stop-optimizer` or `web-app-new-fleet-dashboard`. Do not reuse a flag name after archiving one; LaunchDarkly keeps history and a reused name makes old audit logs confusing.

## Who can create and toggle flags

Any engineer can create a flag for their own service. Toggling a flag in prod for a customer-facing change needs a heads-up in the relevant team channel first, since flag flips are effectively deploys and should follow the same release-window expectations described in `deployment_process.md`, with the exception of an emergency kill switch during an active incident.

## Kill switches

Every new customer-facing feature that touches the dispatch path should ship with a kill switch flag from day one, separate from the rollout flag. The kill switch defaults to "feature enabled," and flipping it off should degrade to the previous known-good behavior without a deploy. This came out of a 2025 SEV1 where a routing optimizer bug required a full rollback because no kill switch existed.

## Rollout process

Typical rollout: internal users only, then 5 percent of accounts, then 25, then 100, holding at each stage for at least a business day while watching Datadog for error rate and latency regressions on the affected service. Jordan Pike's team should know before a flag reaches general availability for any feature that was pitched to customers by name, so sales conversations do not get ahead of what is actually live.

## Targeting

Prefer targeting by account ID over targeting by user ID for fleet-level features, since a fleet manager and their drivers should see consistent behavior. User-level targeting is fine for UI experiments in `web-app` that do not affect data or dispatch behavior.

## Cleanup

A flag at 100 percent for more than 90 days with no plan to remove it should be flagged in the team's next sprint planning for cleanup. Remove the flag and the associated conditional code in the same PR; do not leave dead flag checks in the codebase past the sprint where the flag was fully rolled out. Nina Castellanos's team runs a quarterly audit of flags older than 6 months and files cleanup tickets against the owning team.

## EU considerations

Flags gating anything related to data residency or the eu-west-1 boundary need a review from Tom Brennan's team before toggling, since a misconfigured flag here is a compliance issue, not just a product one.
