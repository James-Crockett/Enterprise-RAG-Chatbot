# Contract review process

Last reviewed: 2026-01-20

Owner: Ruth Adeyemi, General Counsel.

## When legal review is required

Any customer facing agreement needs a legal review before it goes out for signature. This covers new Enterprise contracts, renewals with changed terms, order forms with a discount above 25 percent, and any amendment to the standard Master Services Agreement (MSA).

Growth and Starter tier order forms that use the standard MSA with no redlines do not need legal review. Jordan Pike's sales team can send those directly through DocuSign once Elena Vasquez's finance team confirms pricing.

## Intake

Sales reps open a contract request in Ironclad and attach the order form, any customer redlines, and a short summary of the deal (ARR, term length, discount, any non standard terms the customer asked for). Requests without a deal summary are sent back to the rep before review starts.

Ruth Adeyemi triages new requests each morning and assigns them to herself or an outside counsel firm for the more complex redlines (data residency carve outs, liability cap changes, indemnification changes).

## Review stages

1. Intake and triage (target: same business day)
2. Redline review against the standard MSA (target: 2 business days)
3. Internal sign off if the deal needs Tom Brennan (security terms) or Elena Vasquez (pricing or payment terms) (target: 1 business day, runs in parallel with stage 2 when possible)
4. Final redline sent back to customer's legal team
5. Signature via DocuSign, countersigned by Dana Okafor for Enterprise deals above $250,000 ARR, or by Ruth Adeyemi for everything else

Typical end to end time for a straightforward Enterprise deal is 5 to 7 business days from intake. Deals with data residency or liability negotiations often run 2 to 3 weeks.

## Common redline requests and how they are handled

| Customer ask | Standard response |
|---|---|
| Uptime SLA above 99.95 percent | Declined. 99.95 percent is the ceiling for Enterprise; Starter and Growth stay at 99.9 percent. |
| Liability cap above 12 months of fees | Requires Ruth Adeyemi and Elena Vasquez sign off, case by case. |
| Data residency outside us-east-1 or eu-west-1 | Declined. Only those two regions are supported today. |
| Right to audit security controls | Approved for Enterprise, with 30 days notice and once per year. |
| Termination for convenience | Declined for contracts under 12 months; considered for multi year Enterprise deals. |

## Roles

Ruth Adeyemi owns final legal sign off on every contract. Tom Brennan reviews any security or data processing terms. Elena Vasquez reviews pricing, payment terms, and any custom billing schedule. Jordan Pike or the assigned account executive is the point of contact with the customer throughout.

## Escalation

If a deal is stuck in review and the customer has a hard deadline (end of quarter close, for example), the rep flags the Ironclad request as urgent and pings Ruth Adeyemi directly in the #legal-urgent Slack channel. Urgent flags are reviewed within 4 business hours during office hours, 9am to 5pm local.

## After signature

Signed contracts are stored in Ironclad automatically through the DocuSign integration. The account executive updates the Salesforce opportunity to Closed Won and notifies the assigned Customer Success Manager for Enterprise accounts. Finance is notified automatically through the Ironclad to NetSuite sync for invoicing.
