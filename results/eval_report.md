# AgentRecall Evaluation Report

Seed: `7`
Generated: 2026-11-17T00:29:19.623198+00:00

## A. Multi-session task success (memory ON vs OFF)

| Arm | Tasks | Successes | Success rate |
| --- | ---: | ---: | ---: |
| Memory ON | 40 | 40 | 1.000 |
| Memory OFF | 40 | 0 | 0.000 |
| Delta | 40 | - | +1.000 |

| Intent | Tasks | ON rate | OFF rate | Delta |
| --- | ---: | ---: | ---: | ---: |
| compute_process_window | 8 | 1.000 | 0.000 | +1.000 |
| diagnose_opc_convergence | 4 | 1.000 | 0.000 | +1.000 |
| draft_job_deck | 8 | 1.000 | 0.000 | +1.000 |
| lookup_recipe_parameter | 16 | 1.000 | 0.000 | +1.000 |
| summarize_prior_session | 4 | 1.000 | 0.000 | +1.000 |

## B. Retrieval metrics before vs after curation

| Store | Queries | Precision@5 | Recall@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw ingest | 4 | 0.350 | 0.375 | 0.333 | 0.279 |
| After curation | 4 | 0.950 | 1.000 | 1.000 | 1.000 |

## C. Curation quality

| Metric | Value |
| --- | ---: |
| Dedup precision | 1.000 |
| Dedup recall | 1.000 |
| Contradiction precision | 1.000 |
| Contradiction recall | 1.000 |
| Contradiction F1 | 1.000 |
| Entries before curation | 60 |
| Entries after curation | 40 |
| Store reduction | 33.3% |

Decision counts: `accept`=20, `merge`=10, `skip_duplicate`=10, `supersede`=20

## D. Maintenance effects

| Metric | Value |
| --- | ---: |
| Source documents mutated | 3 |
| Stale entries detected | 3 |
| Stale reasons | {'source_changed': 3} |
| Re-embeds run 1 | 6 |
| Re-embeds run 2 | 0 |
| Documents changed run 1 | 3 |
| Documents changed run 2 | 0 |
| Compacted (retired) | 0 |
| Audit findings | 10 |

| Audit finding type | Count |
| --- | ---: |
| zero_usage_old | 10 |

## E. Latency

| Metric | Memory ON | Memory OFF |
| --- | ---: | ---: |
| ask p50 (ms) | 15.00 | 2.00 |
| ask p95 (ms) | 37.15 | 5.00 |
| asks measured | 120 | 120 |

Mean direct memory recall latency: **8.63 ms** (12 samples).

Reproduce with:

```bash
agentrecall eval run --seed 7
```
