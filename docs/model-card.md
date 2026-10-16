# Model and experiment card

The deliverable is a development experiment system, not a promoted production
credit model. It compares constant-prior prediction, regularized logistic
regression and XGBoost, including frozen historical references.

The intended use is portfolio research, reproducible model diagnosis and method
comparison. Operational lending decisions are outside the evidence established
by this repository. Default horizon and feature timing are undocumented.

The current protocol uses two fixed repeats of three-fold nested stratified
development CV. Each outer base fit uses 960 observations; 320 separate rows fit
a low-complexity monotone sigmoid; another 320 select an F2 threshold; 800 form
the outer development assessment fold. Inner three-fold AP selects settings and
the model-family procedure. The historical 600 test rows are excluded throughout.

The primary planned procedures are `selector_glassbox` and `selector_blackbox`:
their recipe choice uses only inner AP. Full recipe results and the explicitly
adaptive intercept follow-up are reported separately. No outer-winning seed is
promoted as a universal champion. See the current numerical table in
[the technical report](portfolio-audit.md).

Logistic regression provides signed additive base margins and sparsity variants,
but is sensitive to scale/outliers, correlation and intercept treatment. Trees
capture nonlinearities/interactions but show overfitting risk and require more
complex tuning. Class weighting is compared, never mandated. Independent sigmoid
calibration improves probability levels in several settings but cannot create
ranking signal. About 20 calibration events per subset limit certainty.

F2 expresses a preference for recall, not business costs. False positives,
flagged volume and calibration must be considered alongside AP. The new score
estimates have a different fitting budget from the historical experiments;
direct claims of historical-test improvement would be misleading.

Explanations concern base log odds (native/SHAP) or base probabilities (LIME),
while calibrated decisions are shown separately. SHAP covers 800 held-out
development rows per family with three backgrounds. LIME examines 32 stratified
success/error cases per family across three seeds and reports R², local error
and term stability. Its synthetic one-hot/financial profiles can be implausible;
the tree surrogate's limited fidelity is a recorded negative finding.

Saved fold bundles contain `pipeline`, `calibrator`, `partitions`, `recipe` and
seed. Their sibling `results.json` contains raw/sigmoid thresholds. These are
research artifacts tied to a specific fold, not a full-data deployable estimator.
All 126 bundles reproduce their saved assessment outputs. Only trusted local
joblib files should be loaded. Environment versions and original source snapshots
are retained; exact runtime depends on hardware.

Before deployment: obtain provenance/identifiers/timestamps; verify feature
availability; define the target and decision policy; validate on a new cohort;
assess uncertainty including refitting, relevant subgroup behavior, temporal
stability and monitoring ownership. Fairness or legal compliance is not claimed.
