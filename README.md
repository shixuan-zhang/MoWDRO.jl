# MoWDRO.jl

Implementation of **Mo**ment relaxations for data-driven **W**asserstein **D**istributionally **R**obust **O**ptimization problems.
For details on the formulations and asymptotic consistency, please see our [arXiv preprint](https://arxiv.org/abs/2505.19278).


## Prerequisites

* Julia `>= 1.6.7` (see `Project.toml`).
* Python 3 with `pandas` for the `example/summarize_results.py` post-processing script.
* Solvers used by the experiment drivers:
  * The drivers under `example/` currently default to the commercial solvers
    **Gurobi** (LP/MILP and the nonconvex-QCQP baseline via `NonConvex=2`) and
    **Mosek** (SDP moment relaxations) for speed and numerical stability. Each
    requires a valid license and the corresponding Julia wrapper (`Gurobi.jl`,
    `Mosek.jl` + `MosekTools.jl`) installed into the `example` environment.
  * The `MoWDRO` module itself depends only on open-source packages
    (`CSDP`, `ECOS`, `HiGHS`, `SCIP`; see `Project.toml`). If Gurobi/Mosek are
    unavailable, replace the `Mosek.Optimizer` argument to `solve_main_level` /
    `solve_two_stage_copos` in the driver scripts with an open-source SDP
    solver such as `CSDP.Optimizer`, and swap `Gurobi.Optimizer` for an
    LP/QP solver such as `HiGHS.Optimizer`.


## Repository layout

```
src/                         # `MoWDRO` module source
├── MoWDRO.jl                #   module entry point, exports, and defaults
├── types.jl, methods.jl     #   public types and top-level API
├── moment_relax.jl          #   moment-SOS relaxation of the WDRO subproblem
├── bundle.jl                #   level / proximal bundle methods for the main solver
└── baseline/                #   reformulations used only by the baseline comparisons
    ├── copos_form.jl        #     Hanasusanto–Kuhn (2018) copositive-SDP baseline
    └── noncvx_form.jl       #     nonconvex-QCQP baseline
test/                        # module test suite (run via `] test` or `runtests.jl`)
example/
├── experiment_common.jl     # shared TOML/CLI helpers `include`d by every driver
├── summarize_results.py     # per-radius plots and comparison tables from CSVs
├── linear_recourse/         # two-stage linear-recourse experiments
│   ├── production/          #   two-stage multi-product production problem
│   └── allocation/          #   two-stage commodity allocation problem
└── polynomial_loss/         # polynomial-loss WDRO experiments
    ├── portfolio/           #   polynomial-cost portfolio management problem
    └── regression/          #   polynomial mean or quantile regression
```

Per-experiment recipes with the exact commands and TOML variants live in the
per-category READMEs:

* [`example/linear_recourse/README.md`](example/linear_recourse/README.md)
* [`example/polynomial_loss/README.md`](example/polynomial_loss/README.md)


## Quickstart

From the parent directory of `MoWDRO.jl/`, run any driver against its sibling
`.toml`:

```
julia --project=MoWDRO.jl/example \
      MoWDRO.jl/example/linear_recourse/production/production.jl
```

All four drivers follow the same CLI convention (defined in
`example/experiment_common.jl`):

```
julia --project=MoWDRO.jl/example <driver.jl> [<config.toml>] [<output.csv>]
```

* If `<config.toml>` is omitted, the sibling `.toml` next to the driver is used
  (e.g. `production/production.toml`).
* If `<output.csv>` is omitted, a default name of the form
  `result_<name>_<sizes>.csv` is written to the current working directory.

Every driver initializes `Distributed` workers, so add `-p <N>` to run the
per-iteration workload across `N` worker processes:

```
julia -p 4 --project=MoWDRO.jl/example \
      MoWDRO.jl/example/polynomial_loss/regression/regression.jl \
      MoWDRO.jl/example/polynomial_loss/regression/regression_w_noncvx.toml \
      result_regression_noncvx.csv
```


## Configuring an experiment

Every driver parses the same `[experiment]` TOML table via
`example/experiment_common.jl`. The available knobs are:

| Key | Meaning |
| --- | --- |
| `"training sample sizes"` | Explicit list of training-set sizes to sweep. |
| `"training sample sweeps"` | Array of `{start, stop, step}` sweeps (concatenated with the explicit list). |
| `"Wasserstein radii"` | Explicit list of Wasserstein radii to sweep. |
| `"Wasserstein radii sweeps"` | Array of `{start, stop, step}` sweeps. |
| `"Wasserstein order"` | Order `p` of the p-Wasserstein metric (integer, e.g. 2 or 4). |
| `"testing sample size"` | Fresh sample size for out-of-sample evaluation. |
| `"target optimality gap"` | Tolerance passed to the bundle method. |
| `"radius scaling"` | Integer `s` for the `r / (N / N_min)^(1/s)` shrinkage; `s ≤ 0` disables scaling. |
| `"baseline method"` | `"none"`, `"copos"`, `"noncvx"`, or `"all"`. |
| `"random seed"` | Integer for reproducibility, or `"none"` for fresh randomness. |
| `"number of replications"` | Positive integer; each replication draws independent training samples. |
| `"time limit"` | Wall-clock cap in seconds for the bundle method (`-1` disables). |

Problem-specific knobs live under the `[problem]` table and vary by
experiment; see the sample TOMLs beside each driver.


## Post-processing

Turn a result CSV into TikZ/PGFPlots comparison plots and a LaTeX summary
table:

```
python MoWDRO.jl/example/summarize_results.py <result.csv> [<output_dir>]
```

The script aggregates results by `(TRAIN_SIZE, WASS_IDX)` across replications,
emits one `.tex` file per Wasserstein radius with 10–90% bands when replications
are available, and adds computational-time comparison plots automatically when
`CPOS_TIME` or `NCVX_TIME` columns are present in the CSV.
