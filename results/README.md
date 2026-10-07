# Results

`make discovery` writes everything below into `results/discovery/`.

| file | contents |
| --- | --- |
| `overall.csv`, `by_class.csv`, `by_objective.csv`, `by_cell.csv` | hit rate@10, enrichment@10, recovered gain, NDCG@10, AUROC and wells-to-first-hit, averaged over panels. `perfect` is the ceiling; `random` averages 20 permutations per panel |
| `overall_by_family.csv`, `by_class_by_family.csv` | the same panels averaged over *families* instead. The two rules differ wherever a family supplies more than one panel, so both are written |
| `panels_<method>.parquet` | one row per panel: every metric, plus the candidate count, hit count and prevalence that set its ceiling |
| `summary.json` | the tables, the paired family-clustered comparisons against random and against the untrained backbone, and the thresholds the labels were built from |
