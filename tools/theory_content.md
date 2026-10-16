# Theory content for the local report

## 1. What the current work establishes

The project compares two binary classifiers on financial-statement features: a regularized logistic regression called **glassbox**, and an XGBoost tree ensemble called **blackbox**. The outcome is default = 1. The work includes a comparable fixed-parameter baseline and a subsequent imbalance-aware randomized search. The selected experiments are saved separately from the baseline.

There are three distinct questions. **Discrimination** asks whether defaults receive higher scores than non-defaults. **Classification** asks which records cross a chosen decision threshold. **Probability quality** asks whether a numerical score is a reliable estimate of the default frequency. Improvement in one question does not imply improvement in the others.

The selected glassbox has slightly better test ranking and higher recall, but substantially worse probability calibration. The selected blackbox catches more defaults at its validation-selected threshold, yet its test ranking is worse than the original blackbox baseline. Neither result supports an unconditional claim that tuning produced a superior model.

Current calibration is disabled. No lending-cost model, temporal validation, borrower-group validation or population-level explanation-stability study has been established. A label called default does not by itself define the prediction horizon, observation window, legal event definition or availability of features at the decision date. Those are unresolved data-governance questions.

## 2. Probability, odds, scores and decisions

Write x for the available predictors and y ∈ {0,1} for the outcome. The desired conditional default probability is p(x) = P(y = 1 | x). A classifier produces a score p̂(x); its name or range [0,1] does not guarantee that it equals this population probability.

**Odds** are p/(1 − p). **Log odds**, or the logit, are ln[p/(1 − p)]. The inverse transform is the sigmoid: σ(z) = 1/(1 + exp(−z)). For example, a probability of 0.10 has odds 1/9 and log odds approximately −2.197. A contribution of +0.5 in log odds multiplies the odds by exp(0.5); it does not add 50 percentage points to probability.

A binary decision uses a threshold t: predict default when p̂(x) ≥ t. The repository uses exactly this greater-than-or-equal convention. Changing t changes predicted labels without changing the underlying ranking. SHAP and native local contributions here operate on the base model’s log-odds scale, while the decision rule operates on its probability-scale score.

The selected thresholds are 0.566104 for glassbox and 0.056098 for blackbox. Their large difference reflects different score distributions, weighting and selected operating points; it does not show that one model uses a different definition of an actual default. These definitions explain how the implemented model and decision rule fit together. [R1]

## 3. Data scope and class imbalance

The included dataset contains **3,000 rows**, with 2,817 non-defaults and 183 defaults. The event rate is 6.1%, or approximately 15.4 non-defaults for every default. The raw modeling input has 218 columns after excluding the target and configured dropped field. The selected fitted preprocessors each expose 574 transformed columns.

An always-non-default classifier would be 93.9% accurate over the complete dataset, but would have zero default recall. Accuracy alone therefore rewards a trivial majority-class rule. The test partition has 37 defaults and 563 non-defaults; its positive rate is 6.1667%, slightly different from the overall rate because counts are integers.

Class imbalance affects the training loss, the variability of evaluation estimates and the precision achievable at a given recall. Precision also depends on event prevalence in the evaluation population. A change in the future population’s default rate can alter observed precision even when conditional score distributions remain similar.

The current implementation addresses imbalance through stratified partitions, stratified CV, average precision as the selection objective, explicit class-weight alternatives, and validation F2 threshold selection. It does not synthesize observations or discard majority-class records. These are descriptions of the implemented experiment, not evidence that all imbalance-related issues have been solved.

## 4. Splitting, independence and leakage

The same seed-42 stratified split is used for both models: **1,800 training rows with 110 defaults**, **600 validation rows with 36 defaults**, and **600 test rows with 37 defaults**. Stratification approximately preserves the class proportion. It does not make observations independent if the same borrower appears repeatedly or if financial statements share a time-dependent process. [R5]

**Leakage** occurs when information unavailable at the intended decision time influences model fitting or selection. Examples include imputing from the full dataset, selecting features using test labels, learning target-based category encodings outside training folds, or using a financial variable observed after the outcome.

In this repository, learned preprocessing is inside the estimator pipeline. During CV it is refitted separately on each fold’s training rows. Hyperparameters are selected using training CV only. The threshold is chosen from validation predictions. Test labels are used for reporting after these choices are fixed. If calibration is enabled during tuning, validation is further split into disjoint calibration and threshold subsets.

The baseline test results were already available before the tuning work. Although the implemented search does not consume test labels, this is a reused research test partition, not a new external cohort. The next serious comparison should define a fresh assessment set appropriate to borrower identity and time ordering.

## 5. Missingness and imputation

Financial data can be absent for structural, administrative or reporting reasons. Replacing a missing value estimates a usable input; it does not recover the unknown true measurement. Different missingness mechanisms can affect bias and generalization.

The pipeline converts infinite numeric values to missing values, imputes numerical features using the training median, and imputes categorical features using the training most-frequent category. Numerical missingness indicators are enabled. These binary columns allow the model to distinguish an observed median-like value from an imputed one. Indicators are created according to the imputer’s fitted behavior, rather than guaranteeing a separate indicator for every possible future missing value. [R7]

Entirely missing training features are retained through imputation and can subsequently be removed as constant. Every imputation statistic is learned within training data, including within each CV fold. Changing a holdout value must not change those fitted statistics; the project tests this behavior.

An indicator may become predictive because it captures a reporting process rather than an underlying economic mechanism. If that process changes, its association can disappear. Median imputation also compresses uncertainty into a single point estimate. The current work has not compared alternative imputers, modeled missingness mechanisms or established feature-specific business meanings for all missing-value patterns. Those remain potential follow-up analyses.

## 6. Encoding, scaling and feature dimensions

The configured categorical fields are formaWlasnosci_Symbol, schemat_wsk_bilans, schemat_wsk_rzis and pkdKod. One-hot encoding converts observed categories into indicator columns. It avoids imposing a false numerical ordering on category codes. Unknown categories are ignored at transformation time; their group becomes an all-zero representation rather than a newly fitted category. [R6]

For logistic regression, each numerical feature and fitted numerical missingness indicator is standardized using training statistics: z = (x − μ_train)/s_train. This makes numerical coefficient units comparable and changes how regularization acts across differently scaled variables. One-hot indicators are not standardized by this pipeline. XGBoost uses the imputed numerical values without standardization. [R6]

Constant transformed columns are removed by a zero-variance filter. The selected preprocessors contain 574 columns, but dimensions can differ across CV folds when training categories or constant features differ. That is legitimate because each fold learns its own transformation.

Prefixes identify transformed outputs: numeric__ marks a numerical pipeline output, categorical__ marks an encoded category, and missingindicator_ identifies an imputation-related indicator. Feature names in report plots are shortened for readability; the exported CSV retains the full names. No domain meaning is inferred from an unexplained financial ratio code. Scaling is also not robust to every outlier; this implementation has not introduced winsorization or a robust scaler.

## 7. Logistic regression and regularization

Logistic regression assumes an additive margin z(x) = b + Σⱼ βⱼxⱼ, followed by p̂(x) = σ(z(x)). The coefficients are estimated by minimizing binary negative log likelihood, with optional observation weights and a regularization penalty. A positive coefficient raises the model’s log odds as that transformed input increases, holding the other inputs fixed.

**L1** penalization promotes sparse coefficients; **L2** penalization shrinks coefficients more smoothly. C is the inverse regularization-strength parameter: smaller C means stronger regularization. The exact objective normalization and treatment of the intercept depend on the solver. With liblinear, the synthetic intercept feature is also regularized. [R1]

The selected glassbox uses liblinear, L1, C = 0.1, class_weight = balanced, max_iter = 1,000, and tolerance 0.001. The completed search considers C from 0.00001 through 1. This computational bound excludes very weak regularization; it does not establish an optimum outside the searched range.

Advantages are a compact mathematical description, signed coefficients and an exact decomposition of each base-model margin. Limitations are the additive linear form, sensitivity to the feature representation and instability of coefficient allocation among correlated predictors. A standardized coefficient is not automatically a causal effect, a significance test or a universal ranking of feature usefulness. An L1 zero coefficient means zero direct contribution in this fitted representation, not proof that the underlying variable is irrelevant.

## 8. Class weighting and its probability consequences

Class weighting changes how much an error from each class contributes to the fitting objective. In balanced logistic regression, the fitted class weight is n/(2n_c), where n_c is the number of training examples in class c. With 1,800 training rows and 110 defaults, the approximate weights are 8.182 for defaults and 0.533 for non-defaults. Their ratio is about 15.364. [R1]

The logistic search also evaluates unweighted fitting and explicit positive weights of 4 or 8 relative to negative weight 1. XGBoost evaluates no weighting or a positive weight equal to a multiplier times n_negative/n_positive. Multipliers 0.25, 0.5, 1 and 2 are available. The XGBoost ratio is recomputed on each fit’s own labels, including CV folds. [R3]

Why probabilities can shift: for an ideal unconstrained weighted Bernoulli loss, minimizing conditional risk gives q = w₁p / [w₁p + w₀(1 − p)]. Therefore logit(q) = logit(p) + ln(w₁/w₀). This is a derivation under an idealized setting, not a correction validated for these regularized finite-sample models.

Weighting can help a minority-sensitive objective while changing probability interpretation. It is not guaranteed to improve ranking. The selected logistic model is balanced, whereas the selected XGBoost model is unweighted. Independent calibration and observed reliability must be assessed before treating weighted scores as population PD estimates.

## 9. Gradient-boosted trees and XGBoost

A boosted tree classifier constructs a margin as an initial score plus contributions from many decision trees. A tree partitions the input space using feature thresholds and assigns a leaf score. Successive trees are fitted to improve the loss of the current ensemble, using gradient information; XGBoost also uses second-order information and explicit complexity controls. [R2]

The probability-scale score is obtained by applying the logistic link to the final margin. Unlike additive linear logistic regression, the resulting function can represent thresholds, nonlinear responses and interactions without manually adding every interaction term.

This flexibility increases the number of ways to fit noise. Tree depth, leaf constraints, shrinkage, feature and row subsampling, and penalties can restrain complexity. The model can also learn unstable thresholds, proxies for an operational process, or interactions that do not generalize.

The selected blackbox has 250 trees, maximum depth 4, learning rate 0.03, full row and column sampling, minimum child weight 10, gamma 2, L1 leaf penalty 0, and L2 leaf penalty 100. The training objective is binary logistic; histogram tree construction is used. The final tree count is fixed. No early stopping is performed in the current experiment. Better representational capacity is a possible advantage, not evidence that this selected model beats the baseline; its observed test AP is lower.

## 10. What the tree hyperparameters control

**n_estimators** is the number of boosting rounds. **learning_rate** shrinks each new tree’s contribution. These act together: a smaller step can require more trees, but the relationship is empirical and depends on other regularization.

**max_depth** limits tree depth. **min_child_weight** constrains the sum of Hessian-based instance weights in a child; it is not simply a minimum row count. **gamma** requires a minimum loss reduction before a split is accepted. **reg_alpha** and **reg_lambda** penalize leaf weights with L1 and L2 terms. These parameters restrict different parts of the optimization. [R3]

The current search uses tree counts 100, 250, 500 or 800; depths 2, 3, 4 or 6; learning rates 0.01, 0.03, 0.06 or 0.1; child weights 1, 5, 10 or 20; row subsampling 0.6, 0.8 or 1; column sampling 0.5, 0.75 or 1; alpha 0, 0.1, 1 or 10; lambda 1, 5, 20 or 100; gamma 0, 0.5 or 2; and the class-weight options described earlier.

Only 40 combinations were sampled, so most combinations were not evaluated. The winning lambda is at the upper searched value, which describes a boundary result rather than proving that 100 is optimal over every possible strength. Full candidate settings and fold scores are retained in cv_results.csv.

## 11. Cross-validation and search uncertainty

Five shuffled stratified folds are formed within the 1,800 training rows. For each candidate, four folds fit preprocessing and the classifier; the remaining fold evaluates average precision. This is repeated so each training record is held out once. The candidate with the greatest mean fold AP is selected, then fitted on the full training partition. The final classifier does not include validation or test rows in its training fit. [R4] [R5]

The selected glassbox has CV AP 0.1370 with fold SD 0.0358; blackbox has 0.1569 with SD 0.0176. The training prevalence reference is 110/1,800 = 0.0611. The reported standard deviations summarize these five fold scores; they are not confidence intervals and do not account for selecting the best of 40 candidates.

The selected candidate’s own CV mean can be optimistic because it won a search. Nested development validation can assess the full selection procedure, while repeated splits can reveal sensitivity to the partition. These are proposed extensions, not analyses already completed.

A finite randomized search returns the best sampled configuration under its scoring rule and search bounds. It does not guarantee a global optimum or an improvement on every metric. In particular, selecting by AP does not directly optimize calibration, a business loss function, or recall at a specific operational capacity.

## 12. Confusion matrices, recall, precision and F scores

The matrix convention is [[TN, FP], [FN, TP]]. **Recall** = TP/(TP + FN) measures the fraction of actual defaults detected. **Precision** = TP/(TP + FP) measures the default fraction among flagged cases. **Specificity** = TN/(TN + FP); the false-positive rate is 1 − specificity. Accuracy counts correct labels across both classes.

F₁ is the harmonic mean of precision and recall. More generally, Fβ = (1 + β²)PR/(β²P + R). For β = 2, the count formula is F₂ = 5TP/(5TP + 4FN + FP). Its preference for recall is defined by this formula, not by an estimated monetary cost ratio. [R8]

For tuned glassbox, TP = 14, FP = 104, FN = 23, TN = 459. Recall is 14/37 = 37.8%, precision is 14/118 = 11.9%, F₁ is 0.1806, and F₂ is approximately 0.2632. For tuned blackbox, TP = 27, FP = 242, FN = 10, TN = 321: recall 73.0%, precision 10.0%, F₁ 0.1765 and F₂ approximately 0.3237.

These worked examples show why higher recall need not imply higher precision or fewer reviews. Comparing operating points requires specifying the practical cost of missed defaults, false alarms and the available review capacity.

## 13. ROC-AUC, Gini and KS

The ROC curve plots true-positive rate against false-positive rate as the threshold changes. ROC-AUC can be interpreted as the probability that a randomly selected positive receives a higher score than a randomly selected negative, with half credit for ties. A constant score has AUC 0.5 when both classes are present. AUC describes ranking, not probability calibration or a chosen threshold. [R9]

The project defines the credit-model **Gini** as 2 × ROC-AUC − 1. This transformation preserves the ordering of models by AUC. It should not be confused with every other statistic also called a Gini coefficient.

The project’s **KS statistic** is the maximum absolute difference between the empirical score CDFs for defaults and non-defaults, using the two-sample SciPy implementation. It measures score-distribution separation; a high KS alone does not establish accurate PD levels or good decisions at the configured threshold. [R19]

Observed test AUC changes are 0.6727 → 0.7031 for glassbox and 0.7562 → 0.7017 for blackbox. The blackbox baseline has the strongest test discrimination in this comparison. These are single-partition estimates; small differences should not be treated as statistically established without an appropriate paired uncertainty analysis and an assessment set compatible with the intended use.

## 14. Precision-recall curves and average precision

A precision-recall curve traces how the fraction of true defaults among flagged rows changes as more actual defaults are recovered. It concentrates attention on the positive class. Precision is sensitive to prevalence, so results from populations with different default rates require care.

The repository’s metric named pr_auc is **average precision**, AP = Σₖ(Rₖ − Rₖ₋₁)Pₖ, where each precision is weighted by its associated recall increase. This is not trapezoidal integration of a linearly interpolated PR curve. It also is not precision at a single threshold. [R10]

A constant score has AP equal to prevalence under this implementation. On the 600-row test partition the reference is 37/600 = 0.061667. The selected glassbox AP is 0.1295 and selected blackbox AP is 0.1154; both exceed that constant-score reference, but the blackbox baseline AP of 0.1934 is higher than either selected experiment.

AP was the training-CV selection objective because it focuses on positive-class ranking under the observed class imbalance. It does not enforce a minimum recall, precision, subgroup performance, workload or calibration quality. The final operating threshold is therefore a separate selection step. The presentation’s PR plot truncates the display’s y-axis at 0.65 for readability; AP itself is computed from the complete untruncated scores and curve.

## 15. Probability quality: Brier loss and log loss

For binary outcomes, **Brier loss** is mean[(p̂ − y)²]. Smaller values indicate better squared probability forecasts. A score close to 1 for a non-default contributes a large error, as does a score close to 0 for a default. Brier loss combines aspects of reliability, resolution and outcome uncertainty; it is not a pure measure of calibration by itself. [R11]

**Log loss** is −mean[y ln(p̂) + (1 − y) ln(1 − p̂)], with numerical handling of boundary probabilities. It penalizes confident errors strongly. Like Brier loss, it evaluates numerical probabilities rather than only their order or their thresholded labels. [R12]

Selected glassbox test Brier loss rises from 0.0765 to 0.2238. Its score distribution and reliability curve also shift upward, supporting the conclusion that it overpredicts default frequencies after balancing. Selected blackbox Brier loss is 0.05643, close to its baseline’s 0.05651, despite worse test AP.

These results are not contradictory: a score can rank cases reasonably while assigning poor probability levels, or assign conservative probabilities with limited ranking resolution. A meaningful probability benchmark should be defined independently of test outcomes, for example a constant score estimated from training prevalence. Choosing a constant from test labels would be an illustrative retrospective benchmark, not an independent operational prediction rule.

## 16. Calibration and weighted scores

A calibrated score of approximately q should correspond to an observed default fraction near q among comparable scored cases. A **reliability diagram** groups scores and plots average prediction against observed frequency. This repository uses ten quantile bins for the reliability curve, with ties potentially reducing the number of effective bins. [R13]

The implemented ECE is a separate diagnostic: scores are assigned to ten equal-width bins, and each bin’s absolute difference between mean score and default fraction is weighted by the bin’s sample share. Empty bins contribute nothing. ECE is sensitive to binning and sample size; a low value does not prove calibration over all relevant subpopulations.

The pipeline supports an optional sigmoid mapping of clipped base logits, fitted by logistic regression on independent validation labels, or isotonic regression on base probabilities. Both are disabled for the reported experiments. If enabled during tuning, the validation partition is split into separate calibration and threshold subsets, leaving only about 18 defaults in each half. That is a serious data limitation, especially for a flexible calibrator.

Calibration is a proposed next step, not a hidden adjustment in these results. A suitable future design should preserve natural class prevalence in calibration data and separate fitting, calibration, threshold selection and final assessment. Native, SHAP and LIME outputs currently explain the uncalibrated classifier; they would not automatically become exact explanations of a subsequently calibrated score.

## 17. Threshold selection and decision costs

The current search fixes model hyperparameters using training CV, fits the classifier on training data, and selects t to maximize validation F₂. All candidate probability thresholds are considered with the same ≥ rule used in prediction. If two thresholds attain exactly equal best scores, the higher one is selected. [R8]

Glassbox’s validation threshold is 0.566104, yielding validation recall 52.8% and precision 17.1%. Blackbox’s threshold is 0.056098, yielding validation recall 86.1% and precision 12.7%. Those are selection-set scores and are optimistic descriptions of the chosen operating point, not independent estimates of future performance.

For a calibrated probability and a simplified decision problem where only a false positive costs C_FP and a false negative costs C_FN, comparing conditional expected losses gives t = C_FP/(C_FP + C_FN). This is an illustrative derivation under explicit assumptions, not the implemented F₂ rule and not a statement about actual lending costs.

Real decisions may involve review limits, heterogeneous exposures, interventions, abstention or different error costs by case. The repository does not define those quantities. A threshold chosen for recall should therefore be reported with false-positive counts and flagged volume. Tuned glassbox flags 118/600 rows; tuned blackbox flags 269/600. The test curves in the presentation merely describe the already fixed models and thresholds; they do not select new operating points.

## 18. PSI, distribution shift and monitoring

The implemented **population stability index** compares a reference score distribution with a current score distribution. With normalized bin shares r_b and c_b, PSI = Σ_b(c_b − r_b) ln(c_b/r_b). The project uses training-score quantiles to define bins, collapses duplicate edges, and extends outer edges to include out-of-reference-range scores. If the reference is nearly constant, it falls back to fixed probability bins.

Positive additive smoothing prevents zero-share logarithms and is normalized so bin shares still sum to one. The default uses ten bins and epsilon 0.000001. These implementation details matter because different binning and smoothing rules can give different PSI values for the same raw scores.

Selected test PSI values are approximately 0.0262 for glassbox and 0.0154 for blackbox relative to their own training scores. These random-split comparisons can reflect overfitting and sampling differences; they are not direct evidence about time drift. The project does not impose universal alert bands.

Monitoring would need additional views: input distributions, missingness, category frequencies, score distributions, realized outcomes when available, calibration and subgroup performance. Covariate shift concerns inputs; prevalence shift concerns event frequency; concept shift concerns the conditional relationship between inputs and outcome. PSI measures only a chosen binned marginal score comparison. This chapter describes the exact local metric and its interpretation, rather than claiming that a monitoring system is already implemented.

## 19. Interpretation: global, local and faithful to what?

A **global** explanation summarizes a fitted model over some scope. A **local** explanation describes the output for one input. A feature coefficient, a tree-gain importance, a mean absolute SHAP value and a LIME coefficient are different mathematical objects and should not be placed on the same numerical scale. [R20]

In this project, native logistic terms explain the additive base margin exactly. XGBoost gain summarizes training splits globally. SHAP gives local additive attributions and a sample-based global summary. The supplementary LIME demonstration fits an approximate local probability surrogate for one case.

An explanation’s **target** must be explicit: which model version, which row, which output scale, which feature representation, which background and whether calibration is included. Here the common target is the already fitted base classifier in its transformed feature space. Native and SHAP margins are log odds; LIME approximates probability. Calibration is excluded from all three explanation paths.

Faithfulness to a model does not establish that the model describes the economic process correctly. A precise explanation of an inaccurate or biased score is still an explanation of that score. Likewise, explaining one favorable case cannot establish that the model is stable across the population. The report therefore shows observed predictive limitations alongside the explanatory outputs.

## 20. Native logistic coefficients and local terms

For logistic regression, z(x) = b + Σⱼ βⱼxⱼ. The **native local contribution** of feature j is βⱼxⱼ, and the native reference is the intercept b at zero transformed inputs. Summing all contributions and applying σ reproduces the base-model score. The project’s behavioral tests and report audit check this reconstruction.

For standardized numerical features, a one-training-standard-deviation increase changes the margin by βⱼ, conditional on other inputs. The corresponding odds multiplier is exp(βⱼ). For a one-hot feature, the direct coefficient term changes as that indicator moves from 0 to 1. Comparing actual categories can involve changing more than one indicator; the full representation and intercept must be considered.

Missingness indicators also pass through numerical scaling, so their coefficients are not raw unscaled 0-to-1 differences. A numerical coefficient’s sign describes the fitted conditional relation in the representation, rather than an isolated economic intervention.

The presentation ranks coefficients by absolute magnitude and displays their signs. This is a useful inspection of the fitted linear function, but it is not the same as ranking average realized contributions: a large coefficient on a rarely varying feature can have little impact over a particular sample. L1 sparsity and correlated inputs further limit any claim that the coefficient ranking is a universal business-importance ranking. These conclusions follow from the implemented additive model and feature scaling.

## 21. Native tree gain importance

The XGBoost classifier exposes a feature_importances_ vector using gain for this tree model. The report labels it **normalized average split gain**. A feature receives importance when splits using it improve the fitted training objective. The vector is a global description of how the ensemble used features during fitting. [R3]

Gain is not a signed effect. A high value cannot tell whether increasing a financial variable raises or lowers default risk, because a variable can act differently across thresholds and interactions. Gain is also not a per-row explanation and does not decompose an individual probability.

Correlated predictors may substitute for each other, so one can receive split credit while a similar variable receives little. Variables offering more usable candidate splits can have different opportunities to be selected. Importance can change with the training partition, penalties and tree-building choices.

The presentation displays the largest saved normalized gain values for the selected XGBoost model, using the same fitted transformed feature names as SHAP. The full 574-column vector is included in feature_outputs.csv. A zero importance means that feature did not receive gain credit in this fitted ensemble; it does not prove absence of a population relationship or that removing it from every future model is harmless.

## 22. SHAP: additive attribution and the background game

SHAP uses Shapley-value ideas to assign feature credit relative to a specified reference game. For a row x, the additive identity is f(x) = φ₀ + Σⱼφⱼ. In this project, f is the base-model margin in log odds and φ₀ is an expected margin under the chosen background. Positive φⱼ raises the margin relative to that reference; negative φⱼ lowers it. [R14]

The general Shapley formula averages a feature’s marginal contribution across feature subsets: φⱼ = Σ_{S not containing j} [|S|!(M−|S|−1)!/M!] × [v(S ∪ {j}) − v(S)]. The definition of v(S), especially how unobserved features are handled, is essential to what the attribution means. [R14]

For an independent linear setting, the familiar term is βⱼ(xⱼ − E[Xⱼ]); this has a different reference from the native βⱼxⱼ decomposition. Tree SHAP computes attributions efficiently for tree ensembles under its configured dependence assumptions. [R15] [R16]

The sigmoid must be applied after adding the margin terms. In general, σ(E[f(X)]) is not E[σ(f(X))]; the probability shown at a SHAP margin baseline is not automatically the mean background probability. Shapley-style credit also does not turn an observational model into a causal model. These distinctions are necessary to read the local plots correctly.

## 23. SHAP in this repository and how to read its outputs

Glassbox uses LinearExplainer with 100 seed-42 sampled training rows as reference. Blackbox uses TreeExplainer with the same training-background sampling rule, interventional perturbation, and raw model output. The explanation set is 20 deterministically sampled test rows, with the same indices for both models. [R15] [R16]

Interventional perturbation treats unavailable features through the supplied background rather than preserving every observed financial dependency. This is an explanation convention; it is not evidence that independently changing a financial ratio is economically feasible. The selected background can materially affect the baseline and allocations.

**Global bars** rank mean |φⱼ| over the 20 explained rows. Absolute values remove direction, and the sample is too small to establish a full-population importance ranking. **Local waterfalls** start at φ₀, add signed effects, and arrive at f(x); the sigmoid then gives the exported score. Less prominent features are explicitly aggregated into an “other features” term so the total remains intact.

The worked row is chosen from saved positive cases by its blackbox score. That makes it useful for an illustrative comparison, not representative of the population. The numerical audit checks SHAP reconstruction for every saved explanation row, and the appendix prints the selected row’s exact margin components and reconstruction errors.

## 24. LIME: a local surrogate and its diagnostics

LIME fits an interpretable model g near a case x to approximate a more complex model f. Its conceptual objective balances a proximity-weighted fit loss with an explanation-complexity penalty: minimize L(f,g,π_x) + Ω(g). Random perturbations provide the synthetic neighborhood, and the proximity kernel emphasizes nearby samples. [R17]

LIME is implemented in the repository but was not in the saved selected runs’ configured native/SHAP outputs. For this report, a **supplementary one-case LIME demonstration** is generated locally from each saved model, without fitting the classifier again. It uses the same selected case, 100 training-background rows, seed 42, 1,000 perturbations, 15 selected terms, no continuous discretization, and the classifier’s probability output. [R18]

The displayed signed coefficients belong to the locally fitted surrogate in LIME’s internally standardized transformed feature space. They are not exact log-odds contributions and should not be numerically compared with SHAP bars. The local surrogate estimate can differ from the model’s probability and need not obey a classifier’s probability bounds.

The report exports the true score, surrogate local estimate and weighted training R². That R² measures fit on the sampled synthetic neighborhood, not independent predictive performance or explanation stability. Perturbing transformed one-hot columns can produce fractional or inconsistent categories, and independently varying financial ratios can violate real relationships. Repeating explanations across seeds and plausible neighborhoods is a future robustness check.

## 25. Correlation, causality, fairness and explanation stability

Financial quantities often share denominators, accounting identities or economic drivers. When predictors are correlated, several model representations can deliver similar predictions but allocate importance differently. Coefficients, split gain and SHAP credit can therefore vary even when predictive metrics change little.

An attribution answers how a fitted model represents a score under chosen assumptions. It does not establish that changing the highlighted feature would change a borrower’s actual probability of default. A causal or actionable recommendation requires additional assumptions, data and validation beyond these explanations.

Explanation **stability** concerns how outputs change across seeds, background samples, nearby inputs and retrained models. None of these robustness studies is established by a single LIME example or the current 20-row SHAP summary. A useful future study would retain repeated outputs and quantify rank, sign and magnitude agreement under defined perturbations.

Predictive quality and interpretability also do not establish **fairness**. Subgroup event rates, error rates, calibration, selection rates and sample sizes would need examination for a defined population and decision process. The current dataset documentation has not established a complete set of relevant protected attributes or subgroup definitions. No legal compliance or fairness certification is claimed. This is a statement about what the present local evidence does and does not measure, rather than a new compliance framework.

## 26. Reading the current results without overstating them

Selected glassbox test AP is 0.1295 versus baseline 0.1222; ROC-AUC is 0.7031 versus 0.6727. Recall rises from 10.8% to 37.8%, while precision falls from 18.2% to 11.9%. Its weighted scores have Brier loss 0.2238 and a reliability curve showing substantial overprediction. The ranking gain does not make those scores ready for use as calibrated PDs.

Selected blackbox test AP is 0.1154 versus baseline 0.1934; ROC-AUC falls from 0.7562 to 0.7017. Recall rises from 2.7% to 73.0%, but false positives increase from 3 to 242. Much of an operating-point comparison necessarily reflects the changed threshold as well as changed model parameters.

Only 37 test events underlie recall. One additional detected default changes observed recall by about 2.7 percentage points. Five CV fold scores and one test split do not establish stable population superiority. Paired uncertainty estimates, suitable independence assumptions and a fresh validation cohort are needed for stronger conclusions.

The reasonable next experiments are independent calibration of weighted scores, a decision policy defined by costs and capacity, a more robust development-validation design, and explicit time/borrower checks. The baseline remains available. Reporting an unsuccessful or mixed tuning result is part of the evidence; replacing an established baseline should follow a stated objective and validation protocol.

## 27. Reproducibility and what was actually executed

The baseline configuration is config/config.yaml. Selected configurations are config/tuned_glassbox.yaml and config/tuned_blackbox.yaml. Models, predictions and explanations from the selected refits are in artifacts/selected/<model>/. Tuning summaries and candidate-level fold results are in artifacts/tuned/<model>/. The baseline artifacts remain in artifacts/<model>/.

The report builder reads those outputs and verifies core metrics and confusion matrices against exported predictions for all partitions. It checks that the compared test row indices and labels align, that reconstructed loader splits match exported indices, and that native logistic and SHAP decompositions reproduce saved model probabilities. These checks validate internal consistency; they do not prove statistical generalization.

The report loads saved fitted models without retraining. Supplementary LIME is generated for one case, and a state hash checks that explaining does not change the fitted model. Native coefficients, gain and mean absolute SHAP values are exported to reports/data/feature_outputs.csv. LIME weights and diagnostics are exported separately.

Input-file hashes and report-generation package versions are saved in a local manifest. The PDFs, presentation and associated assets stay under the ignored reports/ directory. No model data or report files are uploaded or published. Trusted local joblib files are loaded only as part of this repository’s existing workflow. Exact rerun commands and dependencies are included in reports/README.md.

## 28. Glossary and notation

**AP**: average precision, the project’s PR-AUC metric. **AUC**: area under the ROC curve. **PD**: probability of default; a model score requires calibration evidence before receiving that interpretation. **Margin**: a classifier’s pre-sigmoid score, in log odds for these models.

**TP, FP, TN, FN**: true positive, false positive, true negative and false negative, with default as the positive class. **Precision**: default fraction among flagged rows. **Recall**: fraction of actual defaults detected. **F₂**: precision-recall summary with β = 2; used to select the validation threshold.

**CV**: cross-validation. **Hyperparameter**: a model or training setting selected outside the coefficient/leaf-fitting optimization. **Regularization**: a complexity constraint or penalty. **One-hot encoding**: indicator columns for categories. **Imputation**: replacement of missing entries using a fitted rule.

**SHAP**: additive Shapley-style feature attribution under a specified reference game. **LIME**: a locally fitted interpretable surrogate. **Gain**: a training split-improvement importance in the tree model. **ECE**: the implemented binned expected calibration error. **PSI**: the implemented binned score-distribution comparison.

In the formulas, x denotes an input, y an observed binary outcome, p a population probability, p̂ a model score, t a threshold, βⱼ a logistic coefficient, b an intercept, φⱼ a SHAP attribution and φ₀ its baseline. These symbols describe different mathematical objects; using a common bar-chart style does not make their units interchangeable.
