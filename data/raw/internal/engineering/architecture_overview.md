# Architecture overview

Last reviewed: 2026-02-10 by Marcus Lee

## Summary

Kestrel Route runs as a set of services in the `kestrel/platform` monorepo, deployed to Kubernetes on AWS EKS. The primary region is us-east-1. EU customer data lives in eu-west-1 to satisfy data residency terms in EU contracts. Priya Raman owns the overall architecture; each service has a named owner listed below.

## Services

- `route-api` (Go). Public and partner-facing API for route planning requests. Owner: Marcus Lee's team, on-call rotation `route-api-oncall`.
- `dispatch-worker` (Go). Consumes Kafka topics `dispatch.jobs` and `dispatch.events`, assigns jobs to drivers, writes state to Postgres.
- `billing-svc` (Python). Handles invoicing, usage metering, and Coupa integration for vendor costs. Owner: finance engineering pod.
- `web-app` (React, TypeScript). Fleet manager dashboard.
- `mobile-driver` (React Native). Driver-facing app for iOS and Android.
- `geo-tiles`. Serves map tiles and geocoding lookups, backed by a read replica of the routing graph data.

## Data stores

Postgres 16 on RDS is the system of record for routes, drivers, and billing. Each service that owns data has its own schema; cross-schema joins are discouraged and usually mean a service boundary is wrong. Redis holds session state and short-lived caches, TTL 15 minutes by default. Kafka carries dispatch events and is also used for the billing usage stream, retained 7 days.

## Request flow

A route request comes in through `route-api`, gets validated, and is queued to Kafka. `dispatch-worker` picks it up, computes the assignment (calling `geo-tiles` for distance and ETA data), and writes the result back. `web-app` polls for status over a REST endpoint; the mobile app gets a push through the same event stream via a WebSocket gateway in front of `dispatch-worker`.

## Environments

Three environments: dev, staging, prod. Argo CD manages all deploys from the `main` branch and release tags. Each environment has its own EKS cluster and its own Postgres instance. Staging data is refreshed from a scrubbed prod snapshot every Sunday night by a job Sam Ortiz's team maintains.

## Infrastructure as code

All AWS resources are defined in Terraform, in the `infra/` directory of the monorepo. Changes go through a plan review in the PR before apply. Direct console changes to prod resources are not allowed except during a declared SEV1, and any such change must be reconciled back into Terraform within 2 business days.

## Secrets and access

Secrets are stored in 1Password for local development and in AWS Secrets Manager for running services. LaunchDarkly holds feature flag state, described in `feature_flags.md`. Access to prod AWS accounts requires a request through the ServiceNow access portal, approved by the relevant service owner and by Tom Brennan's team for anything touching customer data.

## Diagram and further reading

A current architecture diagram is kept in Confluence under Engineering > Architecture > Diagrams, updated by whoever last changed a service boundary. If you add a new service, update the diagram in the same PR cycle, not after. For deploy mechanics see `deployment_process.md`. For the API conventions each service should follow, see `api_style_guide.md`.
