# Support escalation process

Last reviewed: 2026-03-10

Owner: support team lead, reporting into Marcus Lee. Security related escalations co-owned with Tom Brennan.

## Ticket intake

All customer issues come in through the help portal as tickets, regardless of tier. Starter and Growth customers get standard support; Enterprise customers with a dedicated CSM can also flag urgent issues through their CSM, who opens a ticket on the customer's behalf and tags it as CSM-flagged.

## Severity levels

| Severity | Definition | First response target | Example |
|---|---|---|---|
| Sev 1 | Product down or unusable for the customer, no workaround | 30 minutes | Live Dispatch board will not load for an entire account |
| Sev 2 | Major feature broken, workaround exists | 2 hours | Proof of delivery photos not uploading, signature still works |
| Sev 3 | Minor issue or bug, limited impact | 1 business day | Cosmetic issue in Analytics report export |
| Sev 4 | Question, how to request, minor enhancement ask | 2 business days | How do I add a new driver to the app |

These targets apply during office hours, 9am to 5pm local across our Austin, Toronto, and Lisbon support coverage. Sev 1 issues are covered outside office hours for Enterprise customers under their SLA.

## Escalation path

A support agent who cannot resolve a ticket within the target time escalates to the on call engineer through the PagerDuty style rotation managed in Jira. Escalation steps:

1. Support agent triages and attempts first line resolution using the internal knowledge base.
2. If unresolved within target time, agent escalates to Tier 2 support (a senior support engineer) and flags the ticket in Jira as escalated.
3. Tier 2 either resolves it or, if it is a product bug, files an engineering ticket and links it to the support ticket, tagging the relevant module owner from the roadmap process.
4. For a suspected security issue (a customer reporting unauthorized access, unexpected data exposure, or an authentication problem) the agent pages Tom Brennan's team immediately regardless of severity level, in parallel with the normal escalation path.
5. Sev 1 issues that are not resolved within 2 hours are escalated to Marcus Lee directly, and a status update goes out to affected customers every 60 minutes until resolved.

## Enterprise specific handling

Enterprise customers have a named CSM who is looped in automatically on any Sev 1 or Sev 2 ticket for their account. The CSM is responsible for the customer relationship and communication cadence during an incident; the support and engineering teams are responsible for the fix. This split avoids a customer getting technical updates from two different people with different framing.

## Root cause and follow up

Every Sev 1 and any Sev 2 that took longer than its target gets a brief root cause note, written by the engineer who resolved it, filed in Jira within 3 business days of resolution. Tom Brennan's team reviews root cause notes monthly for any pattern that suggests a broader reliability or security gap, and feeds that into the quarterly roadmap process input.

## Escalating to leadership

If a ticket involves a customer threatening to churn, a data loss concern, or a PR risk, the support lead notifies Nina Castellanos and Jordan Pike's account owner same day, regardless of the ticket's severity level. This is a judgment call by the support lead, not a strict rule based on severity.

## Metrics

The support team tracks first response time, time to resolution, and reopened ticket rate by severity, reviewed monthly by Marcus Lee. Enterprise SLA compliance (99.95 percent uptime commitment) is tracked separately by Sam Ortiz's infrastructure team and reported to Elena Vasquez's team for any SLA credit calculations.
