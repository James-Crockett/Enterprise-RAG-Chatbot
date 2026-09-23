# Data retention policy

Last reviewed: 2026-02-11

Owner: Ruth Adeyemi, General Counsel. Co-owner: Tom Brennan, Head of Security.

## Purpose

This policy sets out how long Kestrel Systems keeps customer data, internal records, and backups, and who can approve exceptions. It applies to Kestrel Route production data hosted on AWS us-east-1 and eu-west-1, and to internal systems including Salesforce, Jira, Confluence, and Slack.

## Customer data

Active customer data (route plans, dispatch logs, proof of delivery records, driver app telemetry) is retained for the life of the contract. When a contract ends, whether by cancellation or non-renewal, customer data is retained for 90 days to allow for reactivation, export requests, or billing disputes.

After the 90 day window, an automated job in the data platform deletes the customer's records from production databases and object storage. Sam Ortiz's team runs a manual verification pass on the 91st day and files a deletion confirmation in Confluence under Legal > Data Deletion Log.

## Backups

Database backups run nightly and are kept for 35 days on a rolling basis. Backups older than 35 days are overwritten automatically. Because backups roll on a 35 day cycle, a customer's data can persist in a backup for up to 35 days after the 90 day production deletion completes. This is disclosed in the Data Processing Addendum signed with Enterprise customers.

## GDPR requests

EU customers and data subjects can request deletion or export of personal data under GDPR at any time during an active contract. Requests go to privacy@kestrelsystems.com and are logged in Ironclad against the relevant contract record.

| Request type | SLA to acknowledge | SLA to complete |
|---|---|---|
| Data export | 3 business days | 15 business days |
| Data deletion (active contract) | 3 business days | 30 days |
| Data deletion (post contract) | Covered by standard 90 day policy | N/A |

Ruth Adeyemi or her designate signs off on any deletion request that touches data under legal hold, for example an open billing dispute or litigation.

## Internal records

Slack messages are retained per workspace settings, currently 1 year, after which older messages are archived and not searchable. Jira tickets and Confluence pages are retained indefinitely unless a project is explicitly archived and deleted by the project owner. Signed contracts and DocuSign envelopes are retained permanently in Ironclad.

Employee records (HR files, performance reviews, payroll) are retained by Aisha Karim's team per the employee handbook, generally 7 years after termination of employment, in line with Texas recordkeeping norms.

## SOC 2 considerations

This policy is one of the controls reviewed during the annual SOC 2 Type II audit, which renews each March. Auditors ask for evidence that the 90 day deletion job actually ran, so Sam Ortiz's confirmation log in Confluence doubles as audit evidence. Do not delete or edit past entries in that log.

## Exceptions

Any request to deviate from this policy (for example, a customer asking for immediate deletion, or a customer asking Kestrel to hold data longer than 90 days) needs sign off from both Ruth Adeyemi and Tom Brennan. Log the exception and its business reason in the Legal exceptions register before acting on it.

## Related documents

Data Processing Addendum template (Ironclad), Security and Privacy public page, Incident Response Runbook (Security team, Confluence).
