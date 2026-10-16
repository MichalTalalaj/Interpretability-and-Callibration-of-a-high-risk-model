# Portfolio audit: reproducible development evidence

Snapshot: 16 October 2026. All new assessment below is repeated nested **development CV**, excluding the already inspected 600 historical test rows. There is no untouched final assessment cohort. The goal is defensible evidence, not a claim of production readiness.

## Main findings

- The data contain real redundancy, severe missingness, rare categories, infinite values and extreme ratios. Timing, default horizon and borrower identity remain undocumented.
- Clipping numerical extremes is a useful development ablation. Extra feature selection is mixed: group-wise association selection can reduce dimensions, while blanket correlation pruning can discard useful information.
- Both model-family selection procedures beat a constant-score benchmark in these development results. Their AP differences are small relative to fold variation; no statistical or universal superiority is claimed.
- Calibration repairs probability levels but cannot create ranking signal. Strong liblinear regularization also penalizes the intercept; a labeled follow-up compares unpenalized-intercept L2 and weakly penalized-intercept L1.
- LIME is much less faithful/stable for the studied tree model than for logistic regression. All explanations concern the base classifier, with calibrated decisions shown separately.

## Protocol and scope

The historical test membership is frozen in `config/historical_test_indices.csv`, with dataset and index hashes in `config/dataset_identity.json`. Original train + validation membership becomes a 2,400-row development pool containing 146 defaults. No new test set is created by reshuffling.

There are two fixed outer seeds (42, 43), three stratified outer folds per repeat, and three inner folds. In each outer split, 1,600 rows are available for development. These are split into 960 model-fit/search rows, 320 calibration rows, 320 threshold rows; the remaining 800 development rows form the outer assessment fold. Around 58–59 fit events and 19–20 events per calibration/threshold subset make this a conservative but noisy design. Each inner model fit has approximately 640 rows and 39 events.

Preprocessing, raw-field selection, category pooling, clipping, imputation and scaling fit inside each inner training fold. Hyperparameters are selected by mean inner AP. Calibration uses a fixed-C, nonnegative-slope sigmoid on independent, naturally imbalanced calibration rows. F2 thresholds are selected on a third disjoint subset. Both raw and calibrated outputs share exactly the same fitted base classifier. No resampling or early stopping is used.

A nested family selector chooses among eligible recipes using **inner AP only**, before accessing the outer labels. Its outer results assess that selection procedure. The full ablation table is descriptive development evidence and is not used to choose a retrospectively optimal seed. Separate held-out roles use less fitting data than historical experiments, so the new CV scores must not be compared as if they were historical test gains.

## Primary procedure-level results

| Procedure | AP mean | AP SD | ROC-AUC | Brier | Log loss | Precision | Recall | FP / 800 | Flagged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selector_glassbox | 0.1313 | 0.0335 | 0.6585 | 0.0570 | 0.2264 | 8.4% | 67.0% | 362.0 | 49.3% |
| selector_blackbox | 0.1310 | 0.0103 | 0.7063 | 0.0556 | 0.2151 | 10.2% | 65.4% | 290.5 | 40.3% |
| dummy | 0.0608 | 0.0006 | 0.5000 | 0.0571 | 0.2293 | 6.1% | 100.0% | 751.3 | 100.0% |

Values are means over six outer folds (each 800 rows). AP SD is a descriptive fold SD, not a confidence interval; the folds share training observations and each development row is assessed twice. Precision, recall and counts use development-selected F2 thresholds, not a business-cost rule.

## Data diagnosis: verified facts and unresolved hypotheses

- 1,014 infinite numeric cells in the full CSV; the loader converts them to missing values. No exact duplicate records. The historically dropped field has one unique value.
- Eight exact duplicate predictor pairs and 66 development numeric pairs with |Spearman rho| ≥ .98 (at least 100 co-observations). Example exact pairs include `wsk_ROS` / `wsk_rent_sprzedazy` and `wsk_koszty_przychody` / `wsk_poziom_kosztow_operacyjnych`.
- Seven development predictors exceed 95% missingness, including six above 99%. Six observed-value near-constants have a modal share ≥99.5%. No configured predictor is wholly missing or strictly constant over all development rows; fold-level constants still occur.
- `pkdKod` has 368 development categories and 104 singleton levels. The largest absolute ratio is about 1.09 × 10^17. Near-zero denominators are a plausible explanation, but source formulas/units are not supplied, so those values are not declared invalid merely because they are large.
- Two coincident four-field financial profiles cover four non-default rows. The pairs differ on 3 and 17 other fields; matching low-information profiles are not verified borrower IDs. No borrower-aware split can be justified as a complete solution from these signatures.
- No usable timestamp or borrower key is documented. An automated substring name scan flags `podatek_dochodowy` because it contains “date”; manual schema review rejects that substring match as timestamp evidence. The age feature is not an observation date.
- Default = 1 is verified as a binary label, but its horizon, event definition, statement dates, feature availability at decision time, repeated-borrower identity and collection process are unknown. Post-outcome leakage cannot be ruled out from names alone.

Full per-field missingness, cardinality, ranges and unverified availability are recorded in `docs/evidence/portfolio_v1/feature_audit.csv`. Correlation pairs and the frozen historical diagnosis are included alongside it.

## Historical diagnosis (exploratory only)

Original logistic AP was 0.611 in training versus 0.106 in validation. Original XGBoost was 0.991 in training versus 0.186 in validation. Those are large overfitting gaps, visible before consulting test scores. The previous selected XGBoost had mean inner training AP 0.244 versus CV AP 0.157; reducing variance also restricted fitted signal. Its fixed lambda=100, gamma=2 and child weight=10 can produce shallow or constant trees. This is evidence to investigate excessive regularization, not proof of one causal explanation for every score change.

Inspecting the actual old tree structures finds 0/300 constant trees and 10.76 mean leaves for the baseline, versus 40/250 constant trees and 2.46 mean leaves (maximum four) for the selected model. This verifies strong fitted restriction; it does not alone prove that weaker regularization will generalize.

The frozen historical configurations are re-evaluated under the exact same new development protocol as all other recipes. Their new rows (`lr_legacy`, `lr_historical_selected`, `xgb_legacy`, `xgb_historical_selected`) provide controlled comparisons without selecting anything from the historical test.

## All planned ablations

All selected logistic fits finish before the declared 1,000-iteration cap. Candidates that reach the cap are rejected rather than silently scored as successful; this recorded study has zero failed candidates. Per-model iteration counts, intercepts, raw fitted mean scores, fit prevalence and calibration parameters are retained in `model_fit_diagnostics.csv`. The intercept diagnosis is therefore auditable against actual fitted states, not inferred from test losses.

| Recipe | AP | SD | Inner AP | Fit AP | Raw fields | Dimensions | Active coefficients | Search seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.0608 | 0.0006 | 0.061 | 0.061 | 218.0 | 503.0 | — | 0.8 |
| lr_clip | 0.1417 | 0.0366 | 0.169 | 0.337 | 218.0 | 494.5 | 494.5 | 11.5 |
| lr_drop_sparse | 0.1417 | 0.0366 | 0.169 | 0.337 | 215.0 | 488.5 | 488.5 | 21.4 |
| lr_grouped | 0.1510 | 0.0307 | 0.168 | 0.239 | 91.7 | 145.0 | 145.0 | 41.7 |
| lr_historical_selected | 0.1144 | 0.0298 | 0.129 | 0.315 | 218.0 | 503.0 | 45.2 | 1.0 |
| lr_l1 | 0.1347 | 0.0160 | 0.121 | 0.281 | 218.0 | 494.5 | 22.5 | 10.6 |
| lr_legacy | 0.1012 | 0.0285 | 0.113 | 0.760 | 218.0 | 503.0 | 503.0 | 1.5 |
| lr_no_indicators | 0.1455 | 0.0297 | 0.168 | 0.325 | 218.0 | 484.5 | 484.5 | 11.1 |
| lr_rare | 0.1492 | 0.0340 | 0.170 | 0.292 | 218.0 | 251.5 | 251.5 | 22.1 |
| lr_redundancy | 0.1261 | 0.0252 | 0.160 | 0.397 | 160.7 | 435.3 | 435.3 | 37.5 |
| lr_robust | 0.1060 | 0.0196 | 0.108 | 0.197 | 218.0 | 494.5 | 494.5 | 18.4 |
| lr_standard | 0.1093 | 0.0286 | 0.136 | 0.340 | 218.0 | 503.0 | 503.0 | 2.7 |
| lr_weighted | 0.1104 | 0.0258 | 0.141 | 0.298 | 218.0 | 503.0 | 503.0 | 2.9 |
| selector_blackbox | 0.1310 | 0.0103 | 0.171 | 0.907 | 218.0 | 462.0 | — | 55.0 |
| selector_glassbox | 0.1313 | 0.0335 | 0.185 | 0.353 | 198.7 | 387.3 | 387.3 | 23.2 |
| xgb_historical_selected | 0.1182 | 0.0132 | 0.102 | 0.250 | 218.0 | 503.0 | — | 5.1 |
| xgb_legacy | 0.1256 | 0.0154 | 0.149 | 0.999 | 218.0 | 503.0 | — | 15.7 |
| xgb_rare | 0.1384 | 0.0211 | 0.170 | 0.834 | 218.0 | 259.0 | — | 46.5 |
| xgb_search | 0.1386 | 0.0170 | 0.169 | 0.864 | 218.0 | 503.0 | — | 51.9 |
| xgb_weighted | 0.1208 | 0.0121 | 0.151 | 0.974 | 218.0 | 503.0 | — | 63.3 |

Recipe-level runtime is cumulative inner search plus base refit. Raw/sigmoid rows share that cost and must not be added together. Selector rows inherit the winning recipe’s cost and **do not represent the full family search budget**; full procedure cost includes all eligible candidate recipes. Total run time and environment are in completion.json/environment.json.

L2 C values are .0001, .003, .03, .3; clipped L1 uses .003, .01, .03, .1. Numerical clipping is fit-only at 1%/99%. Robust scaling is isolated after clipping. Rare-category cutoffs 5/20, missingness cutoffs .95/.99, correlation cutoffs .90/.98 and supervised raw-field counts 25/75/150 are selected in inner CV. Whole categorical fields are retained or removed together, never individual low-ranked one-hot columns.

The grouped supervised selector ranks raw numerical fields by ANOVA association and categorical fields by a rare-pooled chi-square association heuristic. Scores are **not inferential p-values** and are not interpreted causally. The redundancy reducer greedily retains a deterministic, better-observed representative by absolute Spearman correlation; it preserves complete categorical groups. Clipping, rare pooling and embedded sparsity are separate controlled comparisons.

The eight XGBoost candidates pair (150 trees,.08), (300,.04), (600,.02) with depths 2/3 and moderate L2/child/split controls. Identical candidate paths are used for unweighted, fold-weighted and rare-category variants. This replaces the former broad but sparsely covered Cartesian search. Tree counts are selected by inner validation; no early-stopping set is reused.

## Paired ablation differences

| Challenger | Reference | Mean ΔAP | SD ΔAP | Positive folds | Folds |
| --- | --- | --- | --- | --- | --- |
| lr_weighted | lr_standard | +0.0011 | 0.0103 | 3 | 6 |
| lr_clip | lr_standard | +0.0324 | 0.0341 | 5 | 6 |
| lr_robust | lr_clip | -0.0357 | 0.0350 | 1 | 6 |
| lr_rare | lr_clip | +0.0074 | 0.0313 | 3 | 6 |
| lr_no_indicators | lr_clip | +0.0037 | 0.0208 | 4 | 6 |
| lr_drop_sparse | lr_clip | -0.0000 | 0.0000 | 1 | 6 |
| lr_redundancy | lr_clip | -0.0156 | 0.0213 | 2 | 6 |
| lr_grouped | lr_clip | +0.0093 | 0.0387 | 2 | 6 |
| lr_l1 | lr_clip | -0.0070 | 0.0372 | 2 | 6 |
| xgb_search | xgb_legacy | +0.0130 | 0.0067 | 6 | 6 |
| xgb_search | xgb_historical_selected | +0.0204 | 0.0201 | 5 | 6 |
| xgb_weighted | xgb_search | -0.0178 | 0.0077 | 0 | 6 |
| xgb_rare | xgb_search | -0.0002 | 0.0157 | 2 | 6 |

Every contrast uses paired partitions. Greater AP is better; negative differences are retained. More feature removal does not consistently help. The individual outer ablation rankings are not a fresh confirmation of the best recipe selected after viewing this table.

## Adaptive intercept follow-up

Strong regularization of liblinear’s synthetic intercept can pull even an unweighted score toward .5. This was diagnosed using development fits after the initial plan began. The additional study is therefore explicitly **adaptive development work**, not independent confirmation. The original planned results remain unchanged.

Two L2 recipes use lbfgs (unpenalized intercept), with identical C grids, with and without clipping. A third clipped L1 recipe changes only intercept_scaling from 1 to 100, weakening its intercept penalty. The L2 ablation changes both solver and intercept treatment; its results alone do not isolate the numerical solver from intercept penalization. All six outer partitions, calibration/threshold roles and seeds are identical.

| Adaptive recipe | AP | SD | Brier | Log loss |
| --- | --- | --- | --- | --- |
| lr_lbfgs_clip | 0.1359 | 0.0178 | 0.0567 | 0.2215 |
| lr_l1_intercept | 0.1348 | 0.0201 | 0.0560 | 0.2199 |
| selector_intercept | 0.1323 | 0.0224 | 0.0593 | 0.2507 |
| lr_lbfgs_standard | 0.1155 | 0.0293 | 0.0634 | 0.3213 |

The follow-up improves probability behavior without assigning an arbitrary prior shift. It does not establish that the entire dataset’s signal has been exhausted, nor that the adaptive winner has unbiased final performance.

## Ranking, probabilities and decisions

The sigmoid slope is constrained nonnegative and has an L2 penalty with C=1; the intercept is unpenalized. This keeps complexity low with only about 19–20 calibration events and prevents learning a reverse risk ordering. Zero-slope calibrators become constant. Positive-slope calibration preserves within-fold order except numerical clipping/ties, so AP and optimal F2 decisions often stay unchanged while Brier/log loss change. It does not automatically preserve pooled rankings across different fitted folds.

No isotonic calibration is used with these small subsets. Both raw and sigmoid rows, reliability curves, log loss and Brier are retained. Lower Brier alone is not proof of calibration, and reliability-bin estimates are noisy. F2 expresses a recall preference, not an estimated lending-cost function; flagged volume and false positives remain central outcomes.

## Uncertainty and a signal diagnostic

The fixed predeclared clipped L2 C=.03 benchmark has 3-fold development AP 0.1407; its 99 shuffled-label controls give Monte Carlo p=0.01. This diagnostic supports some association under row-exchangeability, not a performance ceiling, a corrected significance test for all searches, or verified borrower independence. It uses a separate fixed-pipeline CV diagnostic, not the nested-family headline procedure.

| Challenger | Reference | Conditional 2.5% | Conditional 97.5% |
| --- | --- | --- | --- |
| selector_glassbox | dummy | +0.0511 | +0.1150 |
| selector_blackbox | dummy | +0.0565 | +0.1104 |
| selector_blackbox | selector_glassbox | -0.0328 | +0.0326 |
| lr_clip | lr_standard | +0.0102 | +0.0598 |
| lr_grouped | lr_clip | -0.0092 | +0.0328 |

The paired bootstrap resamples the original development rows, stratified by outcome, sharing row multiplicities across both repeats and recipes. It estimates differences in mean outer AP **conditional on the fitted predictions and observed prevalence**. It does not retrain models, correct adaptive design or account for unknown borrower clustering. These intervals must not be presented as statistical superiority of the whole future training procedure.

## Interpretation and stability

Interpretation uses predeclared outer repeat 0/fold 0, with each family’s recipe chosen solely by that fold’s inner AP. SHAP covers all 800 held-out development rows per family, over three 100-row fit-only backgrounds (seeds 42/43/44). No historical test rows enter the new interpretation. Global importances average absolute base-log-odds SHAP; correlated raw fields are also grouped using fit-only |rho|≥.98 connected components, summing signed effects before taking mean absolute magnitude. A connected group may link fields whose mutual correlation is below .98 through a chain.

### glassbox

The explained fold selected `lr_rare`: 235 transformed columns, 49 assessment defaults. All 800 rows are explained; 8 random cases each from TP/FP/FN/TN produce 32 case studies. Native coefficients/gain are exported separately. SHAP maximum probability reconstruction error is 1.11e-16.

Across three backgrounds, mean top-10 Jaccard is 1.000; mean global rank Spearman is 0.970. Background stability is not the same as retraining stability.

LIME has 96 explanations (32 cases × 3 seeds), each using 2,000 perturbations. Median weighted synthetic-neighborhood R² is 0.770, its 10th percentile 0.739; median absolute local-probability error 0.0034, 90th percentile 0.0067. Selected-15-feature Jaccard averages 0.984.

### blackbox

The explained fold selected `xgb_search`: 502 transformed columns, 49 assessment defaults. All 800 rows are explained; 8 random cases each from TP/FP/FN/TN produce 32 case studies. Native coefficients/gain are exported separately. SHAP maximum probability reconstruction error is 2.07e-07.

Across three backgrounds, mean top-10 Jaccard is 0.879; mean global rank Spearman is 1.000. Background stability is not the same as retraining stability.

LIME has 96 explanations (32 cases × 3 seeds), each using 2,000 perturbations. Median weighted synthetic-neighborhood R² is 0.263, its 10th percentile 0.134; median absolute local-probability error 0.0136, 90th percentile 0.0458. Selected-15-feature Jaccard averages 0.587.

### Cross-fit retained/active field stability

| Recipe | Mean Jaccard | Minimum | Fold pairs |
| --- | --- | --- | --- |
| lr_drop_sparse | 0.968 | 0.948 | 15 |
| lr_grouped | 0.374 | 0.151 | 15 |
| lr_l1 | 0.205 | 0.111 | 15 |
| lr_redundancy | 0.829 | 0.708 | 15 |

Jaccard uses retained raw fields for explicit selectors and nonzero raw-field membership for L1. Keeping every field trivially gives high retention overlap; it is not evidence of stable effect magnitudes. Pairwise fold fits share training rows, so these are descriptive stability measurements.

SHAP/native are base log odds, LIME is a base-probability surrogate, and the sigmoid is a separate score mapping. Exact base reconstruction does not establish economic causality. The balanced success/error sample is intentionally not population-weighted; LIME can generate fractional one-hot values and implausible financial combinations. Its weighted R² is in-sample synthetic fidelity, not held-out faithfulness. The full case table, weights and diagnostics are retained. The new presentation also shows the first randomly sampled FN for each family, with its signed SHAP terms and actual raw/calibrated score and threshold; these are errors, not favorable maximum-score examples.

## Validation and reproducibility

All 126 saved base models across the planned and adaptive runs reproduce their assessment predictions. Every fit/calibration/threshold/assessment role is disjoint, every inner fold is fit-only, all metric/count exports are recomputed from predictions, and the original 101 artifact/report files are hash-verified unchanged. Nineteen unit tests include assessment-label isolation, frozen transforms, monotone calibration, clone/weight behavior and a portable exclusion/audit workflow that does not require ignored baseline artifacts.

A cold reproduction reruns two complete recipes across all six outer folds using the portable tracked manifest, comparing raw/sigmoid probabilities, thresholds and labels. The record is `reproduction_check.json`; all original fitting/selection code/configs are archived locally under each run’s source_snapshot. Compact fold-level evidence and the current presentation PDF/PowerPoint are tracked. Fitted models, the theory PDF and other generated reports remain local under ignored directories.

Reproduce the protocol, adaptive follow-up, interpretation and validation with the commands in README.md. Existing output directories are immutable by default; exact resumptions reject changed code/data/configuration. Use a new output directory for a new reproduction. The portable exclusion manifest preserves the original 600 indices verbatim and includes no labels; a CRLF/LF-normalized dataset hash rejects content or ordering changes. The original raw-byte hash is retained too.

Publication verification refits both studies in an export of the staged Git index,
without copying ignored artifacts. `publication_check.json` records the comparison
of selected settings, outer metrics and development predictions. Inner L1 search
scores show small between-refit variation; exact inner-search bitwise reproducibility
is not claimed, and the cause was not isolated. This check uses the existing pinned
local environment, rather than a fresh dependency installation. The recorded SHAP
and LIME study is not repeated as part of this publication check.

## What is still needed before deployment

A documented target horizon/event definition; borrower identifiers and statement/decision/outcome timestamps; verified feature availability and source formulas; appropriate group/time development splits; a new external or future cohort; specified costs and capacity; uncertainty studies that include refitting; subgroup error/calibration analysis where justified; monitoring/label-maturation procedures and operational ownership. None is claimed as implemented by this research benchmark.

## Sources

Method references are linked in the local theory PDF. Key primary sources: [scikit-learn CV](https://scikit-learn.org/1.5/modules/cross_validation.html), [LogisticRegression](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html), [calibration](https://scikit-learn.org/1.5/modules/calibration.html), [XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html), [SHAP paper](https://arxiv.org/abs/1705.07874), [LIME paper](https://arxiv.org/abs/1602.04938). Project measurements are from the supplied CSV and saved experiment outputs.
