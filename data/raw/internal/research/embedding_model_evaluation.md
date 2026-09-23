# Embedding model evaluation for route and address search

Last reviewed: 2026-03-10 by ML team (notes by Priya Raman's org)

## Background

Fleet managers search for addresses, saved routes, and driver notes inside `web-app`. The current search is a Postgres trigram index on `routes.name` and `drivers.notes`, which misses paraphrases and typos badly enough that it shows up in support tickets. This doc tracks our evaluation of embedding models for a semantic search layer to sit in front of the existing search, not replace it outright.

## Eval set

We built a labeled set of 1,200 query and target pairs pulled from 90 days of real search logs across 14 customer accounts, with PII scrubbed before anyone touched the data. Labels came from two engineers on the search team independently marking relevance, with disagreements resolved by a third reviewer. Metric is recall@10 against this set, plus mean reciprocal rank for a secondary view.

## Models tested

| Model | Dim | Recall@10 | MRR | Notes |
|---|---|---|---|---|
| text-embedding-3-small (OpenAI) | 1536 | 0.71 | 0.52 | Baseline, cheapest to run |
| text-embedding-3-large (OpenAI) | 3072 | 0.79 | 0.61 | Best recall, highest per-query cost |
| bge-large-en-v1.5 (self-hosted) | 1024 | 0.74 | 0.55 | Runs on our own GPU nodes, no external API dependency |
| e5-mistral-7b-instruct (self-hosted) | 4096 | 0.77 | 0.58 | Best self-hosted result, heavier inference cost |
| all-MiniLM-L6-v2 | 384 | 0.58 | 0.39 | Fast, clearly too weak on address paraphrases |

## Observations

The gap between `text-embedding-3-small` and `text-embedding-3-large` is bigger than expected, 8 points of recall@10, which suggests the small model is not capturing enough of the address-specific vocabulary in our data, things like abbreviated street types and driver shorthand for depot names.

`bge-large-en-v1.5` self-hosted gets close to `text-embedding-3-large` at less than a third of the parameter count, and it runs fine on the GPU nodes we already have provisioned for other internal tooling. Given no external API dependency and the residency question below, this is currently the leading candidate.

`e5-mistral-7b-instruct` was the best self-hosted option by recall, but inference latency at 4096 dimensions is around 340ms per query on our current hardware, versus 60ms for `bge-large-en-v1.5`. For an interactive search box that gap matters more than the 3 points of recall@10 it buys.

## EU data residency

Any embedding call for EU customer data needs to either run through a self-hosted model in eu-west-1 or use a vendor with an EU-region endpoint under contract. This rules out routing EU queries through a US-region OpenAI endpoint without a separate agreement, and is a big part of why the self-hosted options stay on the table even though the OpenAI models score higher.

## Current recommendation

Ship `bge-large-en-v1.5` self-hosted for the first version, gated behind a LaunchDarkly flag (`web-app-semantic-search`), rolled out to internal accounts first. Revisit `text-embedding-3-large` for US-only accounts if recall in production logs falls meaningfully short of what the eval set predicted. Nina Castellanos's team wants a decision before the Q2 roadmap review.

## Open questions

Whether chunking driver notes by sentence versus by note improves recall meaningfully, we have not tested yet. Also unresolved: how much the eval set, built from existing trigram search logs, is biased toward queries that already worked reasonably well under the old system.
