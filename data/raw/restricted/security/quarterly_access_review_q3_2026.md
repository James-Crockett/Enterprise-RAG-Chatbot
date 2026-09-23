# Q3 2026 quarterly access review

Classification: Restricted

Last reviewed: 2026-09-08 by Tom Brennan (Head of Security) and Sam Ortiz (IT lead)

## Purpose and scope

This review covers access to production systems (AWS, GitHub, Datadog, PagerDuty, production Postgres) and to sensitive internal systems (1Password vaults, Okta admin, the billing-svc admin console) for the quarter ending September 30, 2026. It exists to catch orphaned access from role changes, offboarding, and contractor engagements.

## Method

Sam Ortiz pulled current access lists from Okta, AWS IAM Identity Center, and GitHub org membership on September 1. Tom Brennan cross-checked each list against current headcount and role from the HR system. Discrepancies were sent to the relevant manager for confirmation, with a two week response window.

## Findings

**Orphaned production access.** Two former contractors retained AWS console access past their end date: one contractor who rolled off the geo-tiles project on July 15 still had read access to the us-east-1 production account as of September 1, and one former support contractor retained a Datadog account with access to production logs six weeks after their contract ended. Both were revoked September 2, 2026. Root cause was that offboarding tickets did not include an explicit AWS and Datadog removal step for contractors, only for employees. Sam Ortiz is updating the offboarding checklist to close this gap by September 30.

**Excess Okta admin grants.** Four accounts held Okta super admin rights beyond the two (Sam Ortiz, Tom Brennan) that policy calls for: one was a leftover grant from a 2025 SSO migration project, three were engineers who had been granted admin temporarily for a one time SCIM debugging task in June and never had it revoked. All four were downgraded to standard admin or removed entirely on September 3.

**1Password vault sprawl.** The "Production Secrets" vault has 22 members, more than the 12 who currently need it based on role. Access was originally granted broadly during the initial 1Password rollout in 2024. Tom Brennan is trimming this to role based groups (platform on call, security team, IT) by end of Q3, which will bring membership to roughly 10.

**GitHub org access.** No material findings. Two repos (an old billing-svc prototype and a deprecated internal tool) had broader team access than needed and were moved to a restricted team on September 5.

## Access removed this cycle

| Category | Count removed |
| --- | --- |
| Former contractor AWS access | 2 |
| Excess Okta admin grants | 4 |
| 1Password vault memberships | 8 |
| GitHub repo permissions | 2 |

## Open items for Q4

Tighten the offboarding checklist for contractors (owner Sam Ortiz, due September 30), finish the 1Password vault trim (owner Tom Brennan, due September 30), and introduce a quarterly automated Okta admin report so this doesn't require a manual pull each cycle (owner Sam Ortiz, target October).

## Handling this document

This review names specific individuals' access levels and gaps in our offboarding process. Do not circulate outside the security and IT teams and Ruth Adeyemi, who reviews it for anything with legal or compliance implications, particularly ahead of the March 2027 SOC 2 Type II renewal.
