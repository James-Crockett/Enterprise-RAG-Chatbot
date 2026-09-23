# ETA prediction experiments

Last reviewed: 2026-03-14 by ML team (notes by Priya Raman's org)

## Background

`dispatch-worker` currently estimates arrival times using a static speed table per road class from `geo-tiles`, adjusted by a fixed time-of-day multiplier. Support tickets and Gong call notes from Jordan Pike's team both point to ETA accuracy as a recurring complaint, especially in dense urban routes in Toronto and Lisbon service areas. This doc tracks experiments toward a learned ETA model to replace the static table.

## Data

Training data is 6 months of completed route legs, about 4.2 million records, each with actual travel time, road class sequence, time of day, day of week, weather at departure (pulled from a third-party weather API we already license for route planning), and driver ID. Driver ID is included only as a random-effect feature, not exposed downstream, per a review with Ruth Adeyemi's team on the driver behavior scoring question this could raise.

## Baseline

The current static table gets a mean absolute error of 6.4 minutes across all legs, worse in the two dense urban areas mentioned above at 9.1 minutes MAE, better on highway-heavy long-haul legs at 3.2 minutes MAE.

## Models tried

Gradient boosted trees (LightGBM) on hand-built features: road class mix, historical average speed per segment, time of day, weather, day of week. This got MAE down to 4.8 minutes overall, 6.3 in dense urban areas. Training takes about 40 minutes on a single GPU node and retrains weekly against a rolling 6-month window.

A small feedforward network on the same features performed within noise of LightGBM, 4.9 minutes MAE, with no clear benefit to justify the added deployment complexity, so we did not pursue it further.

A sequence model (small transformer over the segment sequence) got the best offline result, 4.1 minutes MAE overall, 5.4 in dense urban areas, but inference latency was a concern: 85ms per route at typical route length versus 6ms for LightGBM. For a real-time dispatch decision loop calling this per candidate route during optimization, that difference multiplies fast since `dispatch-worker` evaluates dozens of candidate routes per job.

## Current direction

LightGBM is the current production candidate given the latency constraint. We're running a shadow deployment against 5 percent of live traffic in us-east-1 starting the week of March 16, logging predictions without acting on them yet, to compare offline eval numbers against real-time conditions before touching the actual dispatch path.

## Risks and open questions

The training data is drawn from routes actually driven, which biases toward routes the current dispatch system already favors; a genuinely new route through an under-traveled area may get a worse prediction than the offline eval suggests. We do not yet have a good way to measure this bias directly, and it is the most likely blind spot in the current numbers.

Weather data has a known gap for Lisbon, where the licensed API's ground station coverage is thinner than in North America. This shows up as a slightly worse improvement over baseline in that region, roughly 1 point of MAE less improvement than in Austin or Toronto, and hurts calibration for the "confidence interval" version of the model still in early testing.

## Next steps

Compare shadow deployment results after 3 weeks against the offline eval. If it holds up, propose rolling into the live dispatch path behind a kill switch flag, per the standard rollout process in `feature_flags.md`, starting with a single fleet customer who has already asked for improved ETA accuracy in their contract renewal conversation with Jordan Pike.
