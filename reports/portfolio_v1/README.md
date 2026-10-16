# Current portfolio presentation

This version supersedes the earlier presentation for portfolio use. Earlier files remain untouched.
The historical test is excluded; results are repeated nested development CV.

- [Editable PowerPoint](results_presentation.pptx): 35 slides with editable text/tables and actual chart outputs.
- [Presentation PDF](results_presentation.pdf): matching slides generated from the same layout.

The presentation compares regularized logistic regression (glassbox) with XGBoost
(blackbox), including model tradeoffs, data limitations, imbalance handling,
calibration, controlled ablations and native, SHAP and LIME interpretation outputs.
Repeated nested development cross-validation is the assessment protocol; the
previously inspected historical test is excluded from further selection.

Additional generated files are available locally and are excluded from Git:

- `technical_report.pdf` / `.md`: full evidence, hypotheses, failures and tradeoffs.
- `theory_guide.pdf`: current methods, formulas, assumptions and actual result examples.
- `figures/`: report assets; `previews/`: rendered review pages after verification.

Source: `tools/build_portfolio_reports.py` and `tools/portfolio_theory.md`.
Numerical source: `artifacts/portfolio_v1` and the labeled adaptive `portfolio_intercept_v1`.
Input hashes and original code snapshots are retained with experiments.
Only this README and the current presentation PDF/PowerPoint are published from
this directory. The theory guide and technical-report PDFs remain local.
