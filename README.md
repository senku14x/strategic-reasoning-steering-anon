# Strategic Reasoning Steering

Anonymized code and task data for the workshop submission:

> Steering Strategic Reasoning in Chain-of-Thought Models: Structured Geometry,
> Behavioral Shifts, and Interpretive Limits

The repository contains scripts for annotation, activation extraction,
difference-of-means vector construction, intervention generation, OOD transfer,
and analysis. Large model checkpoints and generated activation artifacts are not
included.

## Layout

```text
data/
  final_dataset.json
  ood.json
scripts/
  annotation_v2.py
  annotate_chains.py
  annotate_segments.py
  phase2_geometry.py
  phase2.5_analysis.py
  phase3_run_interventions.py
  phase4_run_ood_transfer.py
```

## Paths

Scripts use repository-relative paths by default. You can override them with:

```bash
export PROJECT_ROOT=/path/to/repo
export DATA_DIR=/path/to/data
export ARTIFACTS_DIR=/path/to/artifacts
export OUTPUTS_DIR=/path/to/outputs
```

API keys are read from command-line arguments or environment variables such as
`OPENAI_API_KEY`. No API keys or generated model outputs are included.
