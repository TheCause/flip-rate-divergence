# Zenodo metadata — FlipRateDivergence null report

## Title
Pre-registered Falsification of Timing-Based Divergence Decomposition in KV-Cache Compression

## Authors
- Régis Rigaud (ORCID: 0009-0008-4157-6336)

## Upload type
Publication → **Report** (or **Preprint** if Zenodo asks for one; this is a
pre-registered null report, not peer-reviewed).

## Description
Use `ABSTRACT.txt` content verbatim.

## Keywords
- KV-cache compression
- LLM inference
- pre-registration
- null result
- flip rate
- StreamingLLM
- H2O
- open science
- reproducibility
- RQZ

## License
Creative Commons Attribution 4.0 International (CC-BY-4.0).

## Access right
Open access.

## Related identifiers (Zenodo Related works field)

Only the semantically exact relation goes in the structured field:

- **Is supplement to**: `10.5281/zenodo.19510356` (LFCM V2)

The full RQZ corpus listing goes in **Description / Notes** (text-free),
because Zenodo's DataCite vocabulary has no "Is sibling of" relation and
"Is part of" requires a parent DOI we don't have. Format used:

```
Part of the RQZ framework (independent research program; each component has its own Zenodo DOI):

Core line:
- LFCM V1 — https://doi.org/10.5281/zenodo.19309379
- LFCM V2 — https://doi.org/10.5281/zenodo.19510356
- FlipRateDivergence (this work)

Related components:
- RBD — https://doi.org/10.5281/zenodo.19019226
- HHA+MIT — https://doi.org/10.5281/zenodo.19019249
- ARH+DH — https://doi.org/10.5281/zenodo.19019269

Additional works:
- Anatomy — https://doi.org/10.5281/zenodo.19021547
- Gap Detection — https://doi.org/10.5281/zenodo.19309874
- KnowledgeMaps — https://doi.org/10.5281/zenodo.19436112
- Missing Attractors — https://doi.org/10.5281/zenodo.19536104
- MIB-V15 — https://doi.org/10.5281/zenodo.18683677
- NLU Manifest — https://doi.org/10.5281/zenodo.18865027
```

This template is reusable for all future RQZ papers.

## Communities
- TODO: pick Zenodo community(-ies) if any (RQZ has no dedicated one yet).

## Contents of the ZIP to upload
```
FlipRateDivergence_Submission/
├── README.md                  # null report (markdown)
├── ABSTRACT.txt
├── zenodo_metadata.md         # this file
├── contract/
│   ├── contract_validation.md       # v0.4 frozen (git tag phase2_contract_v0.4_frozen)
│   ├── phase_0bis_report.md         # pre-check verdict
│   ├── phase_0bis_indices.json      # pre-registered conversation indices
│   └── seed_exploratory.md          # origin of the hypothesis
├── code/
│   ├── benchmark.py
│   ├── metrics.py
│   ├── analyze_phase2.py
│   └── requirements.txt
└── results/
    ├── phase2_final_analysis.json
    ├── phase2_final_analysis.log
    ├── run_all_phase2.log
    ├── env_mistral-7b.json
    ├── env_qwen-7b.json
    ├── env_qwen-14b.json
    └── lfcm_*_phase2_*.json         # 6 raw result files (~87 MB total)
```

## Git reference
Companion GitHub repo (convention per RQZ: one repo per publication):
`TheCause/flip-rate-divergence` (to be created at submission time).

Relevant tags in RQZ_papers:
- `phase2_contract_v0.4_frozen` (freeze point before data collection)
- `flip_rate_divergence_baseline_v1` (code baseline after the h2o_approx fix)

## Version
v1.0 (initial deposit; reports pre-registered null verdict on Phase 2).
