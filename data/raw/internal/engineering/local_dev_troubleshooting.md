# Local dev troubleshooting

Last reviewed: 2026-03-05 by Sam Ortiz

## Setup basics

New engineers run `make setup` from the repo root, which installs Docker dependencies, pulls the base images, and installs pre-commit hooks. This assumes Docker Desktop or an equivalent is already running and that you have VPN access through the company software center with SSO and MFA via Okta Verify, since some setup steps pull from internal package registries.

## "make setup" fails partway through

Most often this is a Docker resource limit issue. `route-api` and `dispatch-worker` together with a local Postgres and Redis need at least 6 GB of RAM allocated to Docker; the default on a fresh Docker Desktop install is often lower. Check Docker's resource settings before filing a ticket.

The second most common cause is a stale 1Password CLI session. Local `.env` population pulls secrets via `op inject`; if that session expired, `make setup` fails silently on a later step rather than erroring clearly at the secrets step. Run `op signin` and rerun `make setup`.

## Pre-commit hooks failing

If `pre-commit` fails with a "hook not found" or version mismatch error, run `pre-commit clean` followed by `pre-commit install` again. This usually means the hook cache is stale after a `.pre-commit-config.yaml` change landed in `main` while you were on an older branch.

## Local services won't start

Check for a port conflict first: `route-api` expects 8080, `dispatch-worker` expects 8090, local Postgres 5432, local Redis 6379. Another local project or a leftover container from a previous session is the usual cause. `docker ps -a` and clean up anything unexpected before `make setup` again.

If Postgres starts but migrations fail against it, check that your local migration state is not ahead of what `main` expects; a `git pull` followed by `make db-reset` (which drops and recreates the local database) resolves this in most cases.

## Kafka locally

Local Kafka runs as a single-broker container via `docker-compose`. If `dispatch-worker` cannot connect, check that the container's advertised listener is set to `localhost` and not the internal Docker network name; this is a known issue on some Docker Desktop networking configurations and is tracked in Jira as a low-priority fix, not expected to be resolved soon.

## VPN and SSO issues

If SSO login through Okta fails locally for a service that checks auth, confirm your Okta Verify push notification actually went through; a stale device registration is common after a phone replacement. Re-register the device from the Okta end-user portal, or file a ticket through the IT portal if that does not work within a few minutes.

## When to ask for help

Post in `#eng-help` with the exact error output and what you have already tried. IT tickets through the ServiceNow portal are answered within 24 hours on business days, but for something blocking your whole day, Slack is faster and more likely to catch someone who has seen the exact issue before. Sam Ortiz's team owns local dev tooling generally and triages `#eng-help` daily.
