# Code review guidelines

Last reviewed: 2026-01-28 by Priya Raman

## Purpose

These guidelines apply to every PR in `kestrel/platform`. The goal is catching real problems before prod, not enforcing personal style preferences. If a review comment is purely stylistic and not covered by the linter, say so and mark it non-blocking.

## Approval requirements

Every PR needs at least one approval before merge. Changes touching `billing-svc` or anything under `infra/` need two approvals, at least one from someone L5 or above. Changes to shared libraries used by more than one service need a review from an owner of each affected service, not just the library's own team.

## What reviewers should check

1. Does the PR do what the description says, and is the description accurate. A PR titled "fix typo" that also changes a timeout value should raise a question.
2. Tests. New logic needs a new test, not just a passing existing suite. A bug fix should include a test that fails without the fix.
3. Error handling. Go services should not swallow errors silently; Python services should not use bare `except:` blocks.
4. Migrations, if present, follow `database_migrations.md`.
5. Anything touching auth, PII, or the EU data boundary gets a closer look and, for anything nontrivial, a ping to Tom Brennan's team before approval.

## Turnaround expectations

First response within 1 business day. If you cannot get to a review in that window, say so in the PR rather than leaving it silent; the author can find another reviewer. Reviewers should not sit on a PR for style nitpicks while blocking a fix that is already past due.

## Author expectations

Keep PRs small enough to review in one sitting where possible. A PR over 400 lines of diff should explain in the description why it could not be split. Draft PRs are fine for early feedback, but mark them draft explicitly so CI resource usage and reviewer attention are not wasted on something not ready.

Respond to every comment, even a one-word "done" or "not doing this because X." An unanswered comment thread is the most common reason reviews drag past a week.

## Merge conventions

Squash merge is the default for feature branches. Commit messages should describe why a change was made, not just what changed; "fix" alone is not an acceptable final commit message. Delete the branch after merge; Argo CD and CI do not need stale branches around.

## Disagreements

If author and reviewer cannot agree, loop in the tech lead for that area rather than letting the PR sit. Marcus Lee is the tiebreaker of last resort for cross-team disagreements, but that should be rare; most disagreements resolve with one more comment thread.

## Bots and automation

The GitHub Actions bot posts a comment with test coverage delta and any new lint warnings. A drop in coverage on new code is a blocking issue unless the reviewer explicitly waives it in a comment explaining why.
