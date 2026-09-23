# Security and privacy at Kestrel Systems

Last reviewed: 2026-03-15

This page explains how Kestrel Systems handles security and data privacy for Kestrel Route customers.

## Certifications

Kestrel Systems maintains SOC 2 Type II certification. The audit is renewed annually each March, and a copy of the current report is available to customers under NDA on request through your account rep or CSM.

Kestrel Systems is GDPR compliant for customers processing personal data of individuals in the EU. A Data Processing Addendum is available and is standard in Enterprise contracts.

## Where your data is hosted

Customer data is hosted on AWS infrastructure. Standard hosting is in the us-east-1 region. Enterprise customers who need EU data residency can have their data hosted in eu-west-1 instead. Data residency is set at contract signing and is not something that changes automatically after the account is live.

## Data retention

While your contract is active, your data is retained for as long as you need it. If your contract ends, your data is retained for 90 days afterward, then deleted from production systems. Database backups are kept on a rolling 35 day cycle, so a small amount of deleted data can persist in backups for up to 35 additional days after the 90 day deletion.

## Access controls

Enterprise customers can use single sign on (SSO) to manage how their team authenticates into Kestrel Route, integrating with common identity providers. Starter and Growth customers use Kestrel Route's own login with multi factor authentication available on all accounts.

Within your account, administrators can set role based permissions to control who can view, edit, or export route and delivery data.

## API security

The public API uses API keys scoped to your account, with a rate limit of 600 requests per minute per key. Keys can be rotated at any time from your account settings. We recommend rotating keys periodically and immediately if you suspect a key has been exposed.

## Incident response

If Kestrel Systems identifies a security incident affecting customer data, affected customers are notified without undue delay, consistent with our contractual and GDPR obligations. Notifications include what happened, what data was affected, and what we are doing about it.

## Reporting a security concern

If you believe you have found a security vulnerability in Kestrel Route, report it to security@kestrelsystems.com. Please do not test for vulnerabilities against live customer accounts without authorization; use a sandbox account or contact us first to arrange a test environment.

## Sub processors

Kestrel Systems uses a limited set of sub processors to deliver the product, including AWS for hosting. A current list of sub processors is available on request from your account rep.

## Questions

For questions about this page, a specific customer's data handling, or a Data Processing Addendum, contact your account rep or CSM, or reach us through the help portal.
