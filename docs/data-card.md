# Dataset card

This is a research dataset with incomplete provenance. The project does not infer
a default horizon or operational feature timing from column names.

| Property | Verified state |
| --- | --- |
| File | Included `zbiór_10.csv`, UTF-8 with BOM |
| Identity | CRLF/LF-normalized SHA-256 plus recorded raw-byte hash in `config/dataset_identity.json` |
| Size | 3,000 rows; 220 columns including target and one constant excluded field |
| Target | `default`, observed values 0/1; 183 positives (6.1%) |
| Predictors | 218 raw configured fields; transformed dimensions depend on fitted encoding/selection |
| Historical exclusion | Original 600 test row indices, copied verbatim without labels |
| Current development pool | Original train + validation: 2,400 rows, 146 defaults |
| Exact record duplicates | None in the supplied CSV |
| Repeated borrowers | Not verifiable: no borrower identifier is supplied |
| Time structure | No usable observation/decision/outcome timestamp is supplied |
| Default horizon/event definition | Undocumented |
| Prediction-time feature availability | Unverified for all configured predictors |
| Collection, sampling and licensing provenance | Not established by the supplied repository |

The development audit finds eight exact duplicate predictor pairs, 66 numerical
pairs with absolute Spearman correlation ≥.98, six fields above 99% missingness,
a seventh above 95%, and six observed-value near-constants. `pkdKod` has 368
development levels and 104 singleton levels. Ratios reach about 1.09×10^17;
the full raw CSV contains 1,014 infinite numerical cells.

Two four-field financial-profile matches involve four non-default rows, with
3 and 17 other fields differing. They are not a verified complete borrower key.
An automated substring scan's `podatek_dochodowy` match is rejected as usable
timestamp evidence. The age field is not treated as a statement date.

The loader converts infinities to missing values; learned preprocessing remains
inside fitting folds. Source formulas and units are needed before declaring
individual extreme ratios erroneous. Post-outcome leakage, repeated borrowers,
nonrepresentative sampling and label errors cannot be excluded by this audit.

See [the audit](portfolio-audit.md) and
[the per-field export](evidence/portfolio_v1/feature_audit.csv). The latter records
observed missingness, cardinality, ranges and explicitly unverified availability.
No record is claimed to be a new untouched assessment observation.
