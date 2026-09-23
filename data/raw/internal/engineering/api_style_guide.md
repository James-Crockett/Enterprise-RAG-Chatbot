# API style guide

Last reviewed: 2026-02-05 by Priya Raman

## Scope

This covers the HTTP APIs exposed by `route-api`, `billing-svc`, and the internal endpoints `dispatch-worker` exposes for the web app. `geo-tiles` follows the same conventions where it applies, though most of its traffic is internal gRPC and is covered separately.

## Resource naming

Use plural nouns for collections: `/routes`, `/drivers`, `/invoices`. Nest resources only one level deep; `/fleets/{id}/drivers` is fine, `/fleets/{id}/drivers/{id}/shifts/{id}` is not. If you need a third level, it usually means the resource deserves its own top-level path with a filter, like `/shifts?driver_id=`.

## Versioning

All public endpoints live under `/v1/`. Breaking changes get a new version prefix; we do not break `/v1/` once a customer is on it. Additive changes, like a new optional field in a response, do not need a version bump. Nina Castellanos's team signs off on anything that changes a public response shape, since partner integrations depend on it.

## Request and response conventions

JSON only, snake_case field names. Timestamps are ISO 8601 in UTC, always with the `Z` suffix, never a bare offset. Money fields are integers in cents, never floats, to avoid rounding surprises in billing-svc.

Pagination uses `limit` and `cursor` query parameters, not page numbers, since route and driver lists change too fast for stable page offsets. Response envelopes include a `next_cursor` field, null when there are no more results.

## Errors

Error responses follow this shape:

```
{
  "error": {
    "code": "route_not_found",
    "message": "Route abc123 does not exist",
    "request_id": "req_9f2c..."
  }
}
```

`code` is a stable machine-readable string, `message` is for logs and debugging, not for showing to end users directly. `request_id` should match the `X-Request-Id` header so support and engineering can trace across systems in Datadog.

## Auth

Public API calls use bearer tokens issued through the customer's Kestrel Route account settings. Internal service-to-service calls go through mTLS inside the cluster; no internal endpoint should trust a bearer token alone as the sole check.

## Idempotency

Any endpoint that creates a resource with side effects, like dispatching a job, must accept an `Idempotency-Key` header and store seen keys for 24 hours. This came out of a 2025 incident where a mobile client retry storm created duplicate dispatch jobs during a network blip.

## Deprecation

Deprecated fields get an `X-Deprecated` header on the response for at least 90 days before removal, and a note in the changelog Nina Castellanos's team maintains. Do not silently drop a field even if usage looks like zero; partner traffic is not always visible in our own telemetry.

## Review

Any new public endpoint needs a review from at least one person outside the authoring team before merge, in addition to the normal code review process, since public API mistakes are expensive to walk back.
