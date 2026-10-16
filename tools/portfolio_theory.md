# Theory for the current portfolio study

## 1. Evidence scope and the prediction problem

The observed label is binary default, but the supplied files do not define its horizon, event rule, observation date or collection process. A supervised classifier can reproduce an association with this label without establishing that the inputs would have been available before the outcome. Technical validation cannot replace missing source documentation.

The complete dataset has 3,000 records and 183 defaults. The already inspected historical 600-row test membership is frozen and excluded from the new work. Its remaining 2,400 rows, with 146 defaults, form the development pool. Repeated nested CV is an internal development assessment, not a newly untouched cohort.

No verified borrower key or usable timestamp supports a complete grouped or temporal split. Coincident financial profiles and an age variable do not establish those identifiers. Stratified random splitting is therefore a documented limitation, not a claim of independence. The study separates verified numerical data problems from hypotheses about source quality and leakage.

## 2. Probability, odds, margins and class decisions

Let x denote available predictors, y ∈ {0,1} the outcome, and p(x) = P(y=1 | x) a population default probability. A fitted classifier produces a score p̂(x); being between zero and one does not guarantee calibration or applicability to a future cohort.

Odds are p/(1−p). The logit is ln[p/(1−p)], and the sigmoid σ(z)=1/(1+exp(−z)) maps a margin back to probability scale. For these classifiers the base margin is in log odds. Adding +0.5 to a margin multiplies odds by exp(0.5); it does not add 50 percentage points to probability.

A decision uses p̂ ≥ t. Ranking, probability levels and the decision threshold are separate objects. A monotone score mapping can preserve within-model order while changing probability quality. SHAP/native terms in this study explain the base margin; LIME approximates base probabilities; independent sigmoid calibration and threshold decisions are recorded separately.

## 3. Logistic regression, penalties and the intercept

Logistic regression fits z(x)=b+Σⱼβⱼxⱼ and p̂=σ(z). Binary negative log likelihood penalizes wrong probability forecasts. L1 and L2 penalties restrict fitted coefficients: L1 can produce exact zeros; L2 shrinks coefficients more smoothly. Smaller C means stronger regularization. A sparse coefficient vector is an implementation property, not proof that discarded financial variables lack economic relevance. [R1]

Intercept treatment matters when the event rate is only about 6%. liblinear regularizes a synthetic intercept feature, so strong regularization can pull b toward zero and unweighted scores toward 0.5. This was observed in development fits and motivated a separate adaptive follow-up. lbfgs leaves the L2 intercept unpenalized. Increasing liblinear intercept_scaling weakens the effective intercept penalty but also changes numerical conditioning.

The L2 follow-up changes solver and intercept treatment together; it cannot identify the numerical solver's isolated causal contribution. The L1 comparison changes only intercept_scaling. Both use matched C grids and partitions. Their development results are retained without turning this adaptive experiment into external confirmation.

## 4. Boosted trees and complexity controls

XGBoost combines decision-tree leaf scores into a margin. Each tree seeks improvements to the current loss using gradient and second-order information. Trees can express thresholds, nonlinear responses and interactions that additive logistic regression cannot represent automatically. This flexibility also creates many ways to fit noise. [R2]

Depth, child-weight constraints, split-gain penalties, leaf-weight penalties and row/column subsampling control different aspects of complexity. min_child_weight refers to Hessian-based weight, not simply a minimum row count. gamma requires loss reduction before splitting; reg_alpha and reg_lambda penalize leaf weights. [R3]

The study pairs learning rates .08/.04/.02 with 150/300/600 trees, using shallow depths and moderate regularization. Equal products of rate and tree count are only approximate search-budget heuristics; they do not make different paths mathematically equivalent. Fixed counts are selected by inner CV. There is no early stopping and therefore no hidden early-stopping subset. Frozen historical configurations provide controlled evidence about overfitting and excessive restriction rather than selecting from historical test scores.

## 5. Missing values and reporting processes

Missing financial values may be structural, administrative, or associated with how a statement was recorded. Median/mode imputation creates usable inputs; it does not recover a verified underlying measurement or establish a missingness mechanism.

The loader replaces infinite numerical values with missing values. Numerical medians, categorical modes, optional numerical missingness indicators and constant-column removal are learned inside every training fold. Six development fields exceed 99% missingness and a seventh exceeds 95%, so almost-unobserved values are explicitly studied rather than silently assumed reliable. [R7]

An indicator can capture predictive reporting behavior and also fail when that behavior changes. The study compares indicators present/absent and sparse-field removal cutoffs selected in inner CV. A cutoff is fit on the training portion; no outer feature statistic or label determines its learned removal set. Negative ablation results remain evidence: removing sparse fields need not improve ranking, even when it reduces dimensions.

## 6. Categorical encoding and complete raw-field groups

One-hot encoding expresses categories as indicators without imposing numerical order. The industry code has 368 development levels, including 104 singleton levels. A large sparse representation can overfit rare categories and vary in size across training folds. The historical 574-dimensional representation is not a required dimension for every new fitted fold.

The category experiment groups rare levels using training counts, with frequency cutoffs 5 or 20 chosen in inner CV for logistic regression. XGBoost uses a fixed rare-level comparison with otherwise identical candidate settings. Unknown levels map to an infrequent bucket when one exists, otherwise use the encoder's declared fallback. [R23]

Feature selection operates on entire raw fields before one-hot encoding. A category field is therefore kept or removed as a group; a low-ranked dummy does not independently delete one category. This preserves a coherent representation and avoids using SHAP or gain from an assessment fold as a selection rule. It does not prove that the encoding is optimal for future unseen industries.

## 7. Extreme values, scaling and clipping

Some observed ratios reach approximately 10^17, and the full CSV contains 1,014 infinite cells. Near-zero denominators are a plausible cause, but unavailable formulas and units prevent declaring each large value invalid. Outlier handling is therefore an explicit modeling ablation rather than an undocumented correction to the data.

StandardScaler uses training means and standard deviations. RobustScaler uses training medians and quantile ranges. A robust scaler does not guarantee numerical robustness for a mostly-zero variable with zero interquartile range; large values can remain influential. [R6]

The clipping recipe learns 1%/99% cutoffs on numerical training values, then applies those fixed cutoffs to held-out rows. It is compared against the same L2 model/C grid without clipping. Robust versus standard scaling is compared after identical clipping. Clipping can erase legitimate rare information and reduce features to constants; observed development gains do not justify labeling all extreme financial observations erroneous or adopting arbitrary domain cutoffs without provenance.

## 8. Feature selection and its validation target

The study compares all raw fields against embedded L1, sparse-field filtering, redundancy reduction and grouped supervised selection. Chosen sparsity, correlation cutoffs, missingness limits and field counts are selected inside inner validation. The outer fold never chooses a feature count.

The grouped supervised method uses univariate numerical ANOVA association and rare-pooled categorical chi-square association as ranking heuristics. It keeps complete raw fields at candidate counts 25/75/150. These scores are not reported as inferential p-values: small counts, multiple comparisons, correlated variables and selection invalidate an easy significance interpretation. [R22]

Univariate ranking can miss interactions or favor redundant fields. L1 can arbitrarily allocate credit among correlated predictors. Removing low-ranked features from one fitted model can therefore harm generalization. Controlled ablations show whether a specific selection procedure helps under a fixed validation design. Fewer dimensions, more stable selected sets and better AP are separate outcomes; the full evidence table reports all of them without assuming they agree.

## 9. Correlation, redundancy and grouped attribution

Eight predictor pairs are exactly duplicated over development rows; 66 numerical pairs have |Spearman rho| ≥ .98 with sufficient co-observations. Correlation is evidence of representational redundancy, but not proof that either field can be removed harmlessly from every model.

The fit-only redundancy selector greedily retains deterministic, better-observed numerical representatives, comparing thresholds .90 and .98 inside inner CV. Categorical fields remain intact. A greedy representative set is not identical to a full correlation-clustering algorithm, and its ordering is recorded by code.

For interpretation, a separate fit-only correlation graph forms connected components. It groups SHAP terms by raw field/component, sums signed local attributions and then computes a mean absolute group magnitude. This differs from summing absolute importances, which can double-count offsetting effects. A connected component may link two fields through a chain even if their direct correlation is below the graph threshold. Group attribution describes model behavior under this grouping convention, not a causal economic factor.

## 10. Nested CV and independent development roles

Two fixed repeats use three stratified outer folds each. Every outer assessment fold has 800 development records. From the other 1,600 rows, 960 are reserved for model fitting/search, 320 for calibration and 320 for threshold selection. Each inner three-fold model fit contains about 640 rows and 39 defaults. [R21]

Preprocessing, selection and class weights are learned within inner training folds. Hyperparameters maximize mean inner AP. A fitted base classifier uses only the 960 model-fit rows. The calibrator never sees its fitting labels, and the threshold subset is disjoint from both. Outer labels assess the fixed pipeline after those decisions.

The family-selector procedure chooses among eligible recipe types using inner AP only. Reporting its outer results avoids retrospectively presenting the best outer ablation as an unbiased winner. This conservative design has a sample-efficiency cost: its fitting budget differs from the historical 1,800-row fit. Repeated development CV still does not erase earlier exposure to development data or provide an external deployment cohort.

## 11. Class imbalance and class weighting

A constant non-default prediction has high accuracy when defaults are rare but zero default recall. Stratification, average-precision scoring, explicit weighting alternatives and separate thresholds address different aspects of that imbalance. None is a universal remedy.

Balanced logistic weights are n/(2n_c). Tree positive weights are either one or the fit-specific negative/positive count ratio, recomputed inside each fold. Weighted and unweighted tree searches use identical candidates, so weighting is a controlled comparison rather than a required feature of a supposedly improved model. [R1] [R3]

For an ideal weighted conditional Bernoulli loss, its minimizer is q=w₁p/[w₁p+w₀(1−p)], so logit(q)=logit(p)+ln(w₁/w₀). This derivation explains why weighting can change probability levels. It is not a validated prior correction for these regularized finite-sample classifiers. The project uses independent natural-prevalence calibration instead of an arbitrary PD shift. Strong intercept penalization is an additional probability issue and can affect unweighted fits too.

## 12. Independent low-complexity calibration

Calibration asks whether score levels match observed default frequencies among similarly scored cases. The current sigmoid maps a clipped base logit z to σ(az+b), constrained to a≥0. It minimizes summed Bernoulli negative log likelihood plus a²/(2C), with C=1 fixed and an unpenalized intercept. Only the independent 320-row calibration subset is used.

There are only around 19–20 calibration defaults. A regularized two-parameter mapping is therefore preferable here to casually fitting a flexible curve. Isotonic calibration is not part of this experiment. A zero fitted slope produces a constant score, explicitly sacrificing resolution rather than learning a reverse ordering.

With positive slope and no additional numerical ties, within-fold ranking is preserved. Calibration cannot create missing ranking signal; it often changes Brier/log loss without changing AP or the maximized F2 decision set. Different fold-specific mappings can change pooled cross-fold rankings. Reliability curves and bin counts must therefore be interpreted with their sampling scope and small event counts. [R13]

## 13. Thresholds, confusion matrices and F2

Default is the positive class. Confusion matrices use [[TN,FP],[FN,TP]]. Precision=TP/(TP+FP), recall=TP/(TP+FN), specificity=TN/(TN+FP), and flagged volume=TP+FP. Under imbalance, precision, false-positive counts and review load are necessary companions to recall.

Fβ=(1+β²)PR/(β²P+R). The current threshold maximizes validation F2, equivalent to 5TP/(5TP+4FN+FP), using the same ≥ comparison as prediction. Exact ties choose the higher threshold. This is a stated recall preference, not an estimated monetary cost ratio. [R8]

If probabilities were calibrated and only false positives/negatives had constant costs, comparing conditional expected losses would yield t=C_FP/(C_FP+C_FN). That illustrative formula is not the project's rule and does not establish actual lending economics. The independent threshold subset has only about 20 events, so its selected operating point is noisy. Its F2 score is a selection score; outer precision/recall/counts assess the resulting policy.

## 14. Ranking measures: AP, ROC-AUC, Gini and KS

Average precision weights each precision value by its associated increase in recall: AP=ΣΔR·P. The repository's historical pr_auc key means AP, not trapezoidal PR integration. A constant score has AP equal to prevalence under this implementation. Precision and AP depend on the evaluation class distribution. [R10]

ROC-AUC measures score ordering across positive-negative pairs, with half credit for ties. A constant score has AUC .5 when both classes are present. The credit-model Gini used here is 2AUC−1. KS measures maximum absolute separation between empirical default/non-default score CDFs. None establishes calibration or a suitable decision threshold. [R9] [R19]

The headline result is the mean outer-fold AP, with descriptive fold SD. Repeat-pooled AP is retained separately because different fitted models can have different score scales. The fixed-family procedures, individual ablations and adaptive intercept follow-up have different selection scopes; their tables must retain those labels instead of collapsing every number into a single leaderboard.

## 15. Probability losses and reliability diagnostics

Brier loss is mean[(p̂−y)²]. Log loss is −mean[y ln(p̂)+(1−y)ln(1−p̂)], with numerical handling at zero/one. Both assess numerical forecasts, and smaller values are better. Brier loss includes reliability, resolution and outcome uncertainty; it is not a pure calibration measure by itself. [R11] [R12]

Reliability curves compare mean scores with observed event fractions in score bins. This study uses quantile bins for stored curves; ECE separately uses equal-width bins and sample-weighted absolute differences. Those binning schemes answer related but not identical questions. Sparse events, ties and empty bins make estimates noisy.

Raw and independently calibrated results use the same fitted base model, isolating the mapping's effect. A very low-complexity prior predictor is an essential probability benchmark. Better AP with much worse Brier is not a complete improvement. Similarly, matching a prior's Brier while offering little ranking resolution does not establish a useful operating policy. All three views—ranking, probability losses and decisions—are reported.

## 16. Uncertainty, dependency and adaptive experimentation

Six outer fold scores are not six independent experiments: training sets overlap, and each development row is assessed twice. Fold SD and repeat ranges are descriptive summaries, not standard-error estimates or automatic confidence intervals. Selecting the best outer result after viewing many recipes introduces additional optimism.

The paired bootstrap resamples original development rows, stratified by outcome, sharing each sampled row's multiplicity across both repeats and recipes. It summarizes differences in mean outer AP conditional on fitted scores and the observed prevalence. It does not retrain, include unknown borrower clusters, model future distribution shift or correct adaptive experiment design.

The intercept study was motivated by development evidence after the initial plan began, and is labeled adaptive. Repeating all seeds and keeping all failures improves transparency; it does not turn an adaptive result into fresh confirmation. A future cohort, or a separately planned fully nested study appropriate to borrower/time independence, is needed for stronger comparative claims. [R21]

## 17. Shuffled labels and the limits of signal claims

The permutation diagnostic fixes a clipped L2 C=.03 pipeline and computes three-fold development AP for the observed labels and 99 shuffled-label datasets. An empirical one-sided Monte Carlo p is (1+number of null scores at least observed)/(1+number of permutations). The resulting resolution is limited by the number of permutations. [R24]

This is a fixed-pipeline diagnostic, distinct from the nested family-selection result. It supports some association when observation-label exchangeability is a reasonable assumption. Borrower independence is unverified, and the diagnostic does not correct every preprocessing/model hypothesis explored in the project.

Neither a modest AP nor a successful permutation diagnostic proves an irreducible performance ceiling. Label quality, unavailable predictors, target timing, heterogeneous reporting and model misspecification remain possible limitations. The defensible conclusion is that useful but limited measured signal exists under the current development protocol; the amount achievable under a verified deployment problem is unknown.

## 18. Native logistic terms and tree gain

For a logistic base margin, b+Σβⱼxⱼ is an exact decomposition. Native local terms are βⱼxⱼ relative to zero transformed inputs; numerical coefficients refer to training-standardized values, and one-hot coefficients refer to indicator coordinates. Missingness indicators are scaled with numerical inputs. Comparing category alternatives can require changing multiple indicators.

Coefficients are conditional features of the fitted representation. Their signs are not causal effects or inferential significance statements, especially with correlated financial quantities and L1 selection. A large coefficient is not necessarily a large average realized contribution.

Tree gain is a normalized global summary of average training split improvements, not a signed local effect. Correlated features can substitute for each other, and a feature unused by one fitted ensemble is not proven economically irrelevant. Native outputs are exported alongside SHAP so their different meanings, units and reference points remain visible. [R1] [R3]

## 19. SHAP, reference games and exact reconstruction

SHAP explains f(x)=φ₀+Σφⱼ under a specified feature-coalition game and background. A Shapley term averages a feature's marginal contribution over subsets with weights |S|!(M−|S|−1)!/M!. The treatment of absent features defines the game; it is not merely a display choice. [R14]

LinearExplainer explains the logistic margin relative to sampled training backgrounds. TreeExplainer uses interventional perturbation and raw output for the tree ensemble. Independent background substitutions need not preserve accounting identities or observed feature correlations. Background choice can redistribute credit. [R15] [R16]

All 800 assessment rows in the predeclared development fold are explained for each selected family model, using three 100-row fit-only backgrounds. Summing SHAP and applying the sigmoid reconstructs the base probability to numerical tolerance. It does not establish causal validity or calibration. In general σ(E[f]) differs from E[σ(f)]. A later calibrator is separately applied to the reconstructed base score; its probability is not obtained by adding log-odds terms directly to probabilities.

## 20. Global importance, cases and explanation stability

Global SHAP bars average |φ| over the specified 800 development rows. They are more extensive than the former 20-row summary, but still describe one predeclared outer fitted model per family. Raw-field and correlated-component aggregation provide alternative views of redundant financial representations.

The case study samples eight TP, FP, FN and TN cases per family with a fixed random seed. This balances successes/errors for inspection; it is intentionally not representative of their population proportions. No maximum-scoring favorable case determines the study.

Across backgrounds, top-10 Jaccard measures overlap of important feature sets, and Spearman compares their importance ranks. Many tied zero importances can make whole-vector rank agreement look stronger than agreement among the leading features. Local sign agreement is also retained. These do not measure retraining stability. Separately, retained/active raw-field Jaccard across all six fitted folds measures how the learned representation changes across training partitions. Stable explanations can describe a weak model, while unstable allocations can occur among correlated variables with similar predictions. Predictive quality and explanation scope must remain separate.

## 21. LIME and measured local fidelity

LIME fits an interpretable surrogate g near x, balancing a proximity-weighted error L(f,g,π_x) against complexity Ω(g). It samples synthetic neighbors and fits a local model rather than producing exact additive SHAP terms. [R17]

The new study generates 32 cases × 3 seeds per family, with 2,000 perturbations and up to 15 selected terms. The same fit-only background is held fixed across seeds to isolate perturbation randomness. LIME targets the uncalibrated base probability and operates in internally standardized transformed coordinates. Signed surrogate coefficients are not log-odds contributions. [R18]

Weighted R² is 1−Σwᵢ[f(xᵢ)−g(xᵢ)]²/Σwᵢ[f(xᵢ)−mean_w(f)]² over synthetic neighbors. Its targets are the classifier's probabilities, not true default labels, and R² can be negative. It measures surrogate fit in that neighborhood, not held-out predictive faithfulness. The local estimate can differ from the model and need not lie within probability bounds. Absolute local error and selected-term Jaccard expose additional failure modes. The studied tree model has weaker R² and term stability than logistic regression. Independently perturbing one-hot indicators and financial ratios can generate implausible profiles, so even high fidelity requires caution.

## 22. Monitoring, PSI and deployment evidence

The legacy metric implementation includes PSI=Σ(c_b−r_b)ln(c_b/r_b), using reference-only score bins, collapsed duplicate edges, infinite outer limits and normalized additive smoothing. A random-split PSI can reflect sampling or overfit; it is not proof about time drift. The new CV study does not claim to have implemented an operational monitoring system.

Real monitoring would need input/missingness/category checks, score distributions, matured labels, discrimination, calibration, workload and defined subgroup analyses. Explanation transparency does not certify fairness or compliance. No verified protected-attribute set, group/time independence or operational intervention process has been established here.

Before deployment, source documentation must define the event/horizon and feature availability; borrower identifiers and timestamps must support appropriate validation; an actually new cohort must assess the chosen process; costs/capacity must define decisions; and ongoing ownership must be specified. These are proposed requirements, not implemented achievements or claims of legal compliance.

## 23. Reproducibility and the meaning of validation

The project freezes dataset content/order and historical exclusion indices, logs complete candidate and fold results, checks convergence, preserves all seeds, and stores independent role memberships. A CRLF/LF-normalized dataset hash supports Git checkouts with different line endings; the original raw-byte hash is also retained. Resuming a changed code/data/configuration identity is rejected. Original executed source/configuration snapshots and environment versions remain with each run.

All 126 new saved base models were loaded and checked against assessment predictions and metrics. Nineteen tests include changing outer labels without changing fit/calibration/threshold outputs, frozen transform statistics, grouped raw-field behavior, monotone calibration and portable exclusions/audit without old ignored artifacts. A cold-refit reproduction of two complete recipes across all six folds matches probabilities, thresholds and labels.

Those checks establish implementation consistency in the recorded environment. They do not prove borrower independence, correct target timing, causal interpretation or future performance. The tracked evidence package supports review without distributing model binaries. Local PDFs and full fitted artifacts remain ignored; no publication is performed by generating them. A configured CI workflow still needs a real remote run before it can be called passing CI.

## 24. Glossary and a defensible project narrative

**AP** is average precision. **AUC** here refers to ROC-AUC. **PD** means probability of default, requiring a documented event horizon and probability validation. **Margin** is the pre-sigmoid model score. **CV** means cross-validation, and **OOF** means a prediction from a model that did not fit that observation's outcome.

**Calibration** concerns probability levels; **threshold selection** concerns decisions. **L1/L2** are regularization forms. **Raw-field selection** preserves complete categorical groups before encoding. **SHAP** is additive attribution under a specified game; **LIME** is an approximate local surrogate; **gain** is a training split-improvement summary. **Jaccard** is intersection size divided by union size.

The portfolio narrative is a diagnosis-driven reproducible study: verify data defects; protect a known historical assessment boundary; compare controlled preprocessing/selection hypotheses; diagnose both overfit and intercept treatment; separate probability mapping from ranking and decisions; measure explanation fidelity; and retain negative results. It is not a claim to have maximized scores, reached a dataset ceiling, passed external validation or produced a deployment-ready credit model.
