# Repository Guidelines

## Project Structure

- `cvr/` contains reusable Python code for text-to-video retrieval, routing, calibration, counterfactual generation, VLM verification, and evaluation metrics.
- `scripts/` contains the numbered pipeline stages, from `00_prepare_msrvtt.py` through `11_plot.py`, and Bash launchers for the model services.
- `configs/default.yaml` defines paths, model endpoints, and experiment parameters. `data/msrvtt/` holds dataset inputs and extracted frames. `outputs/` holds generated rankings, checkpoints, metrics, and plots. Keep both directories out of commits.
- `temp/` contains Stage 0 experiment utilities and the standalone smoke check.

## Setup and Pipeline Commands

Run commands from the repository root.

- `python -m pip install -r requirements.txt` installs Python dependencies.
- `python scripts/00_prepare_msrvtt.py --help` lists the required annotations, split CSVs, and video directory. After preparation, run `python scripts/01_extract_frames.py` to extract frames and `python scripts/02_retrieve.py --split test` to generate candidate rankings.
- Run `bash scripts/serve_vlm.sh` and `bash scripts/serve_llm.sh` in separate terminals when a stage needs model services. Their default ports are 8000 and 8001, respectively.
- `python scripts/10_run_pipeline.py --mode retriever` evaluates the saved test rankings. See `README.md` for training, calibration, routed evaluation, and plotting. Override defaults with `--set key.sub=value`.

## Coding Style

Use four spaces for Python indentation, `snake_case` for functions and modules, and `PascalCase` for classes. Follow the `NN_stage_name.py` pattern for pipeline entry points. Keep reusable logic in `cvr/` and reuse the shared configuration and JSONL helpers in `cvr/utils.py`.

## Validation

Run `python -m compileall -q cvr scripts temp` for syntax checks. Run `python temp/stage0_smoke.py` to exercise data preparation, frame extraction, retrieval, and the yes/no pilot using generated videos and stubbed VLM responses. For changes involving live verification, run `python scripts/03_pilot_yesno.py --split test --n 50` after preparing rankings and starting the VLM service.

## Pull Requests

Name the affected stage or module and summarize the behavior change. Include the validation commands and any dataset split, model, or configuration overrides needed to reproduce the result.
