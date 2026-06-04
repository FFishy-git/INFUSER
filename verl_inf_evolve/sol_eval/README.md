# Sol Eval

`verl_inf_evolve/sol_eval` is the canonical standalone benchmark evaluation pipeline.

It is configured through:
- [verl_inf_evolve/config/sol_eval.yaml](../config/sol_eval.yaml)
- [verl_inf_evolve/config/sol_eval_experiment](../config/sol_eval_experiment)

## Quick Start

Run with the base config:

```bash
python -m verl_inf_evolve.sol_eval.sol_eval
```

Run with an experiment override:

```bash
python -m verl_inf_evolve.sol_eval.sol_eval sol_eval_experiment=debug-run
```

Override checkpoints and benchmarks directly:

```bash
python -m verl_inf_evolve.sol_eval.sol_eval \
  'eval.checkpoints=[0,5,10]' \
  'eval.benchmarks=[supergpqa,gpqa_diamond]'
```

## Notes

- Benchmark names resolve through [verl_inf_evolve/sol_eval/eval_core.py](eval_core.py).
- `humaneval`, `humaneval_full`, `humaneval_plus`, `livecodebench`, and `livecodebench_v5` bypass the native per-question sol_eval path and dispatch to the vendored AZR external runners.
- The external `livecodebench` path routes local models through the existing base/chat detector: base models use a raw-text generic base prompt, while chat models are wrapped with the model tokenizer's chat template.
- The legacy top-level `eval/` package is obsolete and should not be used for new work.
