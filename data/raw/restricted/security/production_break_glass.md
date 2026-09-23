# Production break glass procedure

Classification: Restricted

Last reviewed: 2026-05-12 by Tom Brennan (Head of Security)

## When to use this

Break glass access is for situations where normal role based access through Okta SSO is not enough to resolve an active production incident, for example a full Okta outage, a compromised service account that needs immediate revocation, or a database issue that requires direct RDS access outside the usual read replica tooling. This is not for routine debugging. Using break glass access when normal access would have worked is treated as a policy violation and reviewed by Tom Brennan.

## Who can invoke it

Sam Ortiz, Tom Brennan, Marcus Lee, and the on call platform engineer (per the current PagerDuty rotation) can invoke break glass access. Any use must be declared on the active incident bridge before credentials are retrieved, not after.

## Where the credentials live

Break glass credentials (root AWS account access, the RDS superuser for the production Postgres 16 cluster, and the emergency Okta super admin account) are stored in the "Break Glass" vault in 1Password, restricted to the four people named above. The vault requires a second factor separate from the person's normal Okta MFA. We do not record credential values anywhere outside that vault, including in this document, in Slack, or in incident tickets.

## Procedure

1. Declare on the incident bridge (PagerDuty triggers this automatically for Sev1 and Sev2) that break glass access is being invoked, and state which credential set.
2. Retrieve the specific credential needed from the 1Password Break Glass vault. Do not retrieve more than the incident requires.
3. Perform the minimum action needed to stabilize the incident. Do not use break glass access for anything beyond the immediate fix.
4. Rotate the credential used within four hours of the incident closing. This is Sam Ortiz's responsibility unless he is the one who used it, in which case Tom Brennan rotates it.
5. File a break glass usage report within 24 hours, covering what was accessed, why, what was done, and confirmation of rotation. Reports go to Tom Brennan and are reviewed with Dana Okafor monthly.

## Rotation and audit

Every break glass credential is rotated on this schedule regardless of use: root AWS access quarterly, RDS superuser monthly, emergency Okta super admin quarterly. Sam Ortiz owns the rotation calendar. Rotation dates and confirmation are logged in the security team's internal tracker, not in this document.

If a break glass credential is used outside a declared incident, or a fifth person is found to have accessed the vault, treat this as a suspected credential leak: rotate the affected credential immediately, notify Tom Brennan, and open an incident on the bridge. Do not discuss the details in public Slack channels.

## Usage history, 2026

There have been two invocations in 2026 so far: an Okta outage on February 11 that required the emergency super admin account to restore SSO configuration (used by Sam Ortiz, rotated same day), and a suspected compromised CI runner on June 30 that required root AWS access to isolate the affected VPC (used by Marcus Lee, rotated within two hours). Both were closed with a full usage report on file.

## Review

This procedure is reviewed every six months, or immediately after any use, whichever comes first. Next scheduled review: November 2026.
