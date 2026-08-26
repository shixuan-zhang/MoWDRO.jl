# Linear-recourse experiments

This directory contains two-stage linear-recourse WDRO experiments. Both
drivers follow the shared CLI convention from `../experiment_common.jl` — see
the root [`README`](../../README.md) for the general command shape,
`Distributed`-worker recipe, and full `[experiment]` TOML knob reference.

## Production (`production/`)

Two-stage multi-product production problem adapted from Shapiro–Dentcheva–
Ruszczyński (Ch. 1.3.1, 2009): choose ingredient stock levels `x` up front,
then react to random demand, salvage, and spoilage realizations by selling
products and either salvaging leftover ingredients or purchasing missing ones
at a late-order premium. See the header of `production/production.jl` for the
LP formulation.

Run the default Mo-WDRO-only configuration with 4 workers:

```
julia -p 4 --project=MoWDRO.jl/example \
      MoWDRO.jl/example/linear_recourse/production/production.jl \
      MoWDRO.jl/example/linear_recourse/production/production.toml \
      result_production.csv
```

TOML variants under `production/`:

| Config | Baselines | Notes |
| --- | --- | --- |
| `production.toml` | `"none"` (Mo-WDRO only) | 20 parts, 10 products, `"radius scaling" = 2`, 3 replications. |
| `production_w_copos.toml` | `"copos"` (Hanasusanto–Kuhn 2018 copositive-SDP) | Smaller problem with dual-bound and scaling settings tuned for the baseline. |

## Allocation (`allocation/`)

Two-stage commodity allocation problem adapted from Duque, Mehrotra, and
Morton (2022): allocate supplies `x` across `n` facilities, then serve random
demands at `m` sites while paying a subcontracting penalty for uncovered
demand. See the header of `allocation/allocation.jl` for the LP formulation.

Run it with:

```
julia -p 4 --project=MoWDRO.jl/example \
      MoWDRO.jl/example/linear_recourse/allocation/allocation.jl \
      MoWDRO.jl/example/linear_recourse/allocation/allocation.toml \
      result_allocation.csv
```

The bundled `allocation.toml` sets `"baseline method" = "all"`, so each
`(N, r)` combination is solved by Mo-WDRO alongside both the copositive-SDP
baseline and the nonconvex-QCQP baseline.

## Output columns

The CSVs written by these drivers always contain the Mo-WDRO columns
`TRAIN_SIZE`, `TRAIN_TIME`, `TRAIN_OBJ`, `TEST_MEAN`, `TEST_STD`, `TEST_Q10`,
`TEST_MED`, `TEST_Q90`, indexed by `(WASS_IDX, REP_IDX)`. When baselines are
enabled, additional columns are appended per solver: `CPOS_*` for the
copositive-SDP baseline and `NCVX_*` for the nonconvex-QCQP baseline (each with
matching `OBJ`, `TIME`, `MEAN`, `STD`, and quantile fields). These extra
columns are picked up automatically by `../summarize_results.py`.
