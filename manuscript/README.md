# MNRAS manuscript package

This directory contains a pre-submission MNRAS Paper draft and its generated
figures. It is intentionally marked **not ready for submission** until the
blocking experiments in `PUBLISHABILITY_AUDIT.md` pass.

## Files

- `main.tex` - MNRAS manuscript source.
- `references.bib` - verified literature used by the draft.
- `make_figures.py` - regenerates publication figures from retained result JSON.
- `figures/` - PDF/PNG versions of manuscript figures.
- `PUBLISHABILITY_AUDIT.md` - current gate verdict and required experiments.
- `FULL_TEST_RUNBOOK.md` - frozen multi-seed publication-validation procedure.
- `cover_letter_draft.md` - cover-letter skeleton with AI-use disclosure.
- `build/` - LaTeX build output.

## Rebuild

```bash
python3 manuscript/make_figures.py
tectonic --keep-logs --keep-intermediates --outdir manuscript/build manuscript/main.tex
```

The final verified PDF is copied to `output/pdf/transitflow_mnras_draft.pdf`.

## Required author input

Before submission, replace the placeholder author name, affiliation, e-mail,
funding statement, CRediT roles, repository URL/DOI, and any institutional
acknowledgements. MNRAS also requires the AI-use disclosure in both the paper and
cover letter.
