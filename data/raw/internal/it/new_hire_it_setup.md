# New hire IT setup

Last reviewed: 2026-01-20 by Sam Ortiz, IT lead

## Before day one

IT provisions accounts 3 business days before a new hire's start date, based on the offer data Aisha Karim's team enters into Workday. This includes:

- Okta account creation, with a temporary password sent to the new hire's personal email
- Google Workspace mailbox
- Slack and Jira accounts, added to the relevant team channels and projects
- Laptop ordered and configured: MacBook Pro 14 inch for engineering, MacBook Air for everyone else

## Day one checklist

New hires complete the following in their first day, usually with IT support in person in Austin and Toronto, or over a video call for Lisbon and remote hires:

1. Sign in to Okta with the temporary password and set a permanent one that meets the 14-character policy.
2. Enroll in Okta Verify for MFA on a personal or company phone.
3. Unbox the laptop and confirm it enrolls in Kandji automatically on first boot.
4. Install the VPN client from the company software center and confirm a test connection.
5. Set up 1Password and import any shared vaults relevant to the role.
6. Complete the security awareness training module assigned automatically in the HR portal.

## Access by role

| System | Who gets it |
| --- | --- |
| GitHub | Engineering, plus Priya Raman's and Marcus Lee's direct reports |
| Salesforce | Sales, under Jordan Pike, and customer success |
| ServiceNow admin | IT team only, everyone else gets standard requester access |
| Coupa | Managers and above, for approving purchases |
| Expensify | Everyone |

Role-based access is assigned automatically from the Workday job profile. If something is missing on day one, file a ticket in the IT portal and it is treated as urgent for the first week.

## Buddy and manager responsibilities

The hiring manager confirms the new hire's calendar, team channel invites, and any project-specific tool access that falls outside the default role template. IT does not automatically know about a tool a specific team uses unless the manager requests it in advance.

## First week follow-up

IT checks in with every new hire around day 5 to confirm:

- VPN and MFA are both working without issues
- The laptop has finished its initial security baseline install
- No pending access requests remain open

## Common first-week issues

**Okta Verify push not arriving.** Usually a notification permission issue on the phone. Reinstall the app and re-enroll the factor.

**VPN connects but internal tools time out.** Often a stale Okta session. Sign out of Okta fully, reconnect the VPN, then sign back in.

**Laptop stuck on the Kandji enrollment screen.** Restart and try again on a stable network. If it happens twice, file an urgent ticket, since this usually means the device profile did not register correctly with Kandji before shipping.

## Remote and international new hires

New hires in Lisbon or fully remote get their laptop shipped 5 business days ahead of their start date to allow for customs. IT schedules a video setup call for day one instead of an in-person session.
