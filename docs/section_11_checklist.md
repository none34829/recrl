# Section 11: Safety, Bias, and Toxicity Audit Checklist

## Pre-RunPod Setup

- [ ] Set up Perspective API key in Google Cloud Console
  - Create a project in Google Cloud Console
  - Enable Perspective Comment Analyzer API
  - Generate API key
  - Save key securely in password manager (NEVER commit to Git)

## RunPod Setup

- [ ] Set environment variable for Perspective API key:
  ```bash
  echo 'export PERSPECTIVE_KEY="YOUR_KEY_HERE"' >> ~/.bashrc
  source ~/.bashrc
  ```

- [ ] Install audit dependencies:
  ```bash
  conda activate rec
  pip install perspective-api-client==1.1.0 gender-guesser==0.4.0 python-Levenshtein==0.23.0 reportlab==4.1.0
  ```

## Local Implementation Status

- [x] Created code/audit package structure
- [x] Implemented toxicity.py for Perspective API integration
- [x] Implemented bias.py for Gini and gender parity metrics
- [x] Implemented privacy.py for Levenshtein distance checks
- [x] Created master audit runner (run_audit.py)
- [x] Added policy thresholds (docs/policy_thresholds.yaml)
- [x] Set up GitHub Actions CI workflow (.github/workflows/safety.yml)
- [x] Created PDF summary generator (generate_pdf_summary.py)

## RunPod Execution

- [ ] Run audits on all datasets:
  ```bash
  conda activate rec
  for d in books ml25m steam; do
    python code/audit/run_audit.py --dataset $d
  done
  ```

- [ ] Generate PDF summary:
  ```bash
  python code/audit/generate_pdf_summary.py
  ```

- [ ] Commit reports to Git:
  ```bash
  git add docs/safety_report_*.json docs/safety_summary.pdf
  git commit -m "Add safety reports"
  git push
  ```

## Verification

- [ ] Confirm all reports exist (3 JSON files)
- [ ] Review PDF summary for all datasets
- [ ] Verify CI passes (all metrics within thresholds)
- [ ] Check that all audits cover the required dimensions (toxicity, bias, privacy)

## Expected Outputs

- Three JSON reports in docs/safety_report_*.json
- PDF summary in docs/safety_summary.pdf
- Passing GitHub Actions workflow for safety checks
