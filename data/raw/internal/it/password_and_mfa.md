# Password and MFA policy

Last reviewed: 2026-02-10 by Tom Brennan, Head of Security

## Overview

Kestrel Systems uses Okta as the single sign-on provider for all internal tools, including Slack, Jira, Confluence, GitHub, Google Workspace, Workday, ServiceNow, and Salesforce. Employees log in once through Okta and get access to every connected app without a separate password for each one.

## Password requirements

Okta enforces the following rules on the primary account password:

| Rule | Requirement |
| --- | --- |
| Minimum length | 14 characters |
| Complexity | At least one letter, one number, one symbol |
| Reuse | Cannot match your last 10 passwords |
| Expiration | 180 days |
| Lockout | 5 failed attempts locks the account for 15 minutes |

Do not reuse your Kestrel password on any personal account. IT cannot recover a forgotten password over Slack DM or email. Use the "Forgot password" link on the Okta sign-in page, which sends a reset link to your registered recovery email.

## Multi-factor authentication

MFA is mandatory for every employee and contractor. Kestrel uses Okta Verify as the required MFA method. Install it on your phone during new hire setup or from your phone's app store if you switch devices.

1. Open the Okta end-user dashboard.
2. Go to Settings, then Extra Verification.
3. Select Okta Verify and follow the QR code prompt to link your phone.
4. Confirm the test push notification succeeds before closing the setup screen.

SMS-based MFA is disabled company-wide because it is not resistant to SIM-swap attacks. If Okta Verify is not available, ask Sam Ortiz or the IT portal to issue a temporary hardware token instead.

## Lost or replaced phone

If you lose the phone enrolled in Okta Verify, do not wait for the next login prompt to sort it out. File a ticket in the IT portal marked "urgent, MFA device lost" and it routes to on-call IT. On-call IT verifies your identity over a video call (camera on, government ID visible) before resetting the factor, since a phone call alone is not enough for this kind of request.

## VPN and remote access

Remote access to internal systems requires the VPN client, installed from the company software center, plus SSO and MFA on every connection. The VPN session times out after 12 hours of inactivity and requires you to reauthenticate.

## Suspicious activity

Report any of the following to Tom Brennan's team through the IT portal:

- An Okta push notification you did not request
- An email asking you to enter your Kestrel password outside okta.com
- A new device or location shown in your Okta sign-in history that you don't recognize

Security reviews the report within 4 business hours during office hours and resets credentials immediately if compromise looks likely.

## FAQ

**Can I use a password manager?** Yes, 1Password is provisioned for every employee and is the only approved manager for storing Kestrel credentials.

**Does MFA apply to the office badge system?** No, badge access is separate and is managed by facilities, not IT.

**What happens after 180 days if I ignore the reset prompt?** Okta locks the account until the password is changed. There is no grace period extension.
