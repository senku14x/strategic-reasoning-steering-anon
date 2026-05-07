# Strategic Reasoning Steering

> Steering Strategic Reasoning in Chain-of-Thought Models: Structured Geometry,
> Behavioral Shifts, and Interpretive Limits

The repository contains reference scripts for annotation, activation extraction,
difference-of-means vector construction, intervention generation, OOD transfer,
and analysis. These scripts are intended to document the computational logic of
the paper rather than provide a fully reproducible artifact: large model
checkpoints, generated activations, raw model outputs, and API-side annotation
artifacts are not included.

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

The `phase*` script names reflect the experimental pipeline stages used in the
paper. They are kept to make the code easy to map onto the reported methods and
appendix results.

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
