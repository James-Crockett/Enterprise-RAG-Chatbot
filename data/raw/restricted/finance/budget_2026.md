# 2026 operating budget

Classification: Restricted

Last reviewed: 2026-02-10 by Elena Vasquez (CFO)

## Purpose

This document sets the approved 2026 operating budget for Kestrel Systems by department. Figures are in USD thousands unless noted. The board approved this budget on 2026-01-28. Any variance over 8 percent against a quarterly line requires CFO sign off before spend continues.

## Headline numbers

Total planned opex for 2026 is $54,200k against a revenue target of $78,000k (ARR, year end). That implies an operating margin target of roughly 30 percent, up from 24 percent in 2025.

| Department | 2026 budget | 2025 actual | Change |
| --- | --- | --- | --- |
| Engineering | 19,400 | 16,800 | +15% |
| Sales and marketing | 15,600 | 13,200 | +18% |
| Customer success | 6,100 | 5,400 | +13% |
| G&A | 7,300 | 6,900 | +6% |
| Security and IT | 3,400 | 2,600 | +31% |
| Facilities | 2,400 | 2,300 | +4% |

## Engineering detail

Engineering is split across three cost centers: platform (route-api, dispatch-worker, billing-svc), mobile and web, and infrastructure. Infrastructure jumped this year because of the eu-west-1 buildout finishing in Q1 and higher Datadog usage tied to the new geo-tiles service. Marcus Lee owns this line and reviews it monthly with Priya Raman.

Headcount funded in this budget: 18 net new engineering hires, weighted toward Q1 and Q2 so the platform team can support the enterprise pipeline Jordan Pike's team is building. Do not communicate specific headcount numbers outside the leadership team before the Q1 board meeting.

## Sales and marketing detail

The increase funds four new enterprise account executives and a doubling of the paid pipeline budget for Growth tier customers. Jordan Pike has flagged that win rate on Enterprise deals fell to 22 percent in Q4 2025 from 31 percent a year earlier, and this budget assumes that trend reverses by Q3. If it does not, marketing spend gets pulled back first.

## Security and IT

The 31 percent increase covers the penetration test remediation program (see security/pentest_findings_2026.md), a second full time security engineer reporting to Tom Brennan, and a planned move of secrets management fully onto 1Password Teams with SSO enforcement. This line item is one reason this document is restricted: the remediation timeline and spend level would tell a reader a lot about where our defenses are thin right now.

## Contingency and risk

We hold a $2,100k contingency reserve, unallocated, for one of three scenarios: a SOC 2 finding that requires unplanned remediation, a security incident requiring outside forensics support, or a slower than planned Enterprise ramp. Elena Vasquez controls release of this reserve; nothing above $150k moves without her and Dana Okafor both signing off.

## Compensation assumptions

This budget assumes a 4 percent average merit increase pool effective April 2026 and holds the comp bands defined in compensation_bands_l6_and_exec.md flat for the year except for a planned equity refresh cycle in Q3. Do not share the merit pool percentage with managers before the April rollout; People team will communicate through the normal channel.

## Review cadence

Elena Vasquez reviews actuals against this budget with department heads on the second Monday of each month. Material changes go to Dana Okafor and the board finance committee before they take effect.
