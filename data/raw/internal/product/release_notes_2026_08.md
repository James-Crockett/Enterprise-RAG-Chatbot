# Release notes, August 2026

Last reviewed: 2026-08-29

Owner: Nina Castellanos, Head of Product. Prepared with Marcus Lee's engineering team.

## Summary

This release focused on Live Dispatch performance and two customer requested Analytics reports. It shipped in 3 batches over the month rather than one single release, which is now the norm for larger quarters.

## Route Planner

No functional changes this month. The team fixed a bug where multi stop routes with more than 40 stops occasionally recalculated stop order after a manual reorder was saved. This affected a small number of Enterprise customers with very large routes and is now resolved.

## Live Dispatch

Dispatch board load time improved for fleets over 100 vehicles. Boards that previously took 6 to 8 seconds to load now load in under 2 seconds, based on internal testing against a 150 vehicle test account. This came from a database query change in how driver status is fetched, not a UI change, so the board looks the same.

Added a new filter to show only vehicles with an active exception (late stop, route deviation, vehicle offline for more than 10 minutes). Requested by 4 separate Enterprise accounts through Productboard over the past two quarters.

## Driver App

Fixed an issue on Android where the app would occasionally fail to update stop status if the driver's phone lost signal mid update. The app now queues the status update locally and retries once signal returns, instead of silently dropping it. This shipped to the Android app store on August 12 and to iOS (no change needed there, issue was Android specific) not applicable.

## Proof of Delivery

Added support for capturing a signature and a photo on the same stop, rather than requiring one or the other. Previously drivers had to choose. This was the top requested Proof of Delivery item in Productboard for two consecutive quarters, mostly from field service customers who want both for insurance purposes.

## Analytics

Two new reports, both requested by Enterprise customers during quarterly business reviews:

- On time delivery rate by region, broken out by the customer's own regional tags rather than just by depot.
- Driver idle time report, showing time between stops that is not attributable to drive time or planned breaks.

Both reports are available now to all Growth and Enterprise customers. They are not available on Starter, since Analytics is a Growth and Enterprise module.

## API

No breaking changes. Added a new read only endpoint, GET /v1/exceptions, that returns active route exceptions, matching the new Live Dispatch filter above. Existing rate limit of 600 requests per minute per API key applies to this endpoint as well.

## Integrations

Samsara telematics sync interval improved from every 5 minutes to every 90 seconds for vehicle location data. Salesforce, NetSuite, and Shopify integrations had no changes this month.

## Known issues going into September

The exception filter in Live Dispatch does not yet support saving as a default view per user, this is planned for September. A small number of customers using Firefox have reported the Analytics report export to CSV sometimes truncating at 10,000 rows, engineering is investigating and this will be tracked separately from this release.

## Questions

Customer facing questions about this release should go through the normal support channel (help portal ticket). Internal questions can go to #product-updates in Slack or directly to Nina Castellanos.
