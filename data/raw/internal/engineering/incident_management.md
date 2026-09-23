# Incident management

Last reviewed: 2026-02-20 by Tom Brennan

## Severity levels

| Severity | Definition | Ack time | Example |
|---|---|---|---|
| SEV1 | Customer-facing outage or data loss risk | 15 minutes, page immediately | route-api returning 5xx for all traffic |
| SEV2 | Degraded service, workaround exists | 30 minutes | Dispatch delays of several minutes fleet-wide |
| SEV3 | Minor, internal, or single-customer edge case | Next business day | One customer's report export is malformed |

Severity can change during an incident. If a SEV2 gets worse, declare it a SEV1 and repage rather than waiting for someone to ask.

## Declaring an incident

Anyone can declare. Create a Slack channel named `#inc-YYYYMMDD-short-name`, for example `#inc-20260218-route-api-5xx`. Post in `#incidents` linking the channel. Page the relevant on-call rotation through PagerDuty if it is not already paged.

Every incident channel gets an incident commander, usually the on-call engineer unless someone more senior takes over. The IC's job is coordinating, not necessarily fixing. Assign a separate scribe if the incident runs past 30 minutes so the IC can stay focused on the response.

## During the incident

Post status updates in the incident channel at least every 30 minutes for a SEV1, every hour for a SEV2, even if the update is "still investigating." Customers on affected accounts get a status page update from Jordan Pike's team or the on-call IC if sales is unavailable; do not wait for sales to be reachable before updating the status page.

Do not debug in DMs. Anything relevant to diagnosis or decisions belongs in the incident channel so the postmortem has a record.

## Resolution

An incident is resolved when the customer-facing symptom is gone, not necessarily when the root cause is fixed. Note the resolution time in the channel and update the status page. If a workaround was applied instead of a real fix, open a follow-up Jira ticket immediately and link it in the channel before closing.

## Postmortems

SEV1 and SEV2 incidents need a postmortem within 5 business days. Postmortems are blameless: they describe what happened and what will change, not who made a mistake. Use the Confluence postmortem template under Engineering > Incidents > Templates.

A postmortem needs, at minimum: timeline, customer impact (which accounts, how long, what they saw), root cause, and a list of follow-up actions with owners and due dates. Follow-up actions get tracked in Jira with the label `postmortem-action` so Priya Raman's team can audit completion monthly.

## Common mistakes

Declaring severity too low because the person paged does not want to wake others up. When in doubt, declare the higher severity and downgrade later; downgrading is easy, escalating late is not.

Closing the incident channel before the postmortem is done. Leave incident channels archived, not deleted, so the history is searchable later.

## Contacts

Tom Brennan owns this process and reviews all SEV1 postmortems personally. Questions about severity calls or escalation paths go to him or to Marcus Lee.
