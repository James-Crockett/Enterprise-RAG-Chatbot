# Deployment process

Last reviewed: 2026-01-14 by Marcus Lee

## Release train

Prod deploys happen Tuesday and Thursday, 10:00 to 15:00 Central. Deploys outside that window need a documented reason and sign-off from the on-call VP Engineering delegate. Code freeze runs December 15 through January 5 each year; the only changes allowed during freeze are hotfixes approved by Marcus Lee directly, logged in the `#eng-freeze-approvals` Slack channel.

## Pipeline

1. Open a PR against `main` in `kestrel/platform`. CI runs through GitHub Actions: lint, unit tests, and a build of the affected service's container image.
2. Get at least one approval per `code_review_guidelines.md`. Two approvals required for changes touching `billing-svc` or anything in `infra/`.
3. Merge to `main`. GitHub Actions builds and pushes the image to ECR, tagged with the short SHA.
4. Argo CD picks up the new image reference for the dev environment automatically, on a 3 minute sync interval.
5. Once dev looks healthy (check Datadog dashboards for the service, no new error budget burn), promote to staging by updating the image tag in the corresponding `infra/argocd/staging/<service>.yaml` file and merging that PR.
6. Run the staging smoke suite (`make smoke-staging` from the repo root). It takes about 8 minutes.
7. Promote to prod the same way, editing `infra/argocd/prod/<service>.yaml`. This PR needs the release-window approval described above.
8. Watch the Argo CD UI for the sync to go healthy, then watch Datadog for 15 minutes post-deploy before calling the deploy done.

## Rollback

Argo CD keeps the last 10 sync revisions per app. To roll back, either revert the image tag PR and let Argo CD resync, or use `argocd app rollback <app-name> <revision>` directly from a bastion with prod access. A direct rollback via the CLI still needs a follow-up PR within 1 business day to keep git and the cluster state consistent.

## Feature-gated releases

Anything risky should ship behind a LaunchDarkly flag rather than a held branch. See `feature_flags.md` for flag naming and cleanup rules. Merging to `main` and deploying to prod does not mean a feature is live for customers if it is still behind a flag at 0 percent.

## Database changes

Migrations deploy as part of the normal service pipeline but follow their own review path, described in `database_migrations.md`. A migration PR needs a second reviewer from the data or platform team regardless of team size.

## Who to contact

Sam Ortiz's team owns the GitHub Actions runners and ECR. Marcus Lee owns the release calendar and freeze exceptions. Priya Raman is the escalation point if Argo CD itself is down or misbehaving across multiple apps at once, since that usually means a cluster-level issue rather than a single service problem.

## Common failures

A stuck Argo CD sync is usually a resource quota issue in the target namespace or a bad readiness probe on the new pod. Check `kubectl describe pod` in the namespace before opening a ticket. If the sync is stuck for more than 20 minutes, page the on-call engineer for that service rather than waiting it out.
