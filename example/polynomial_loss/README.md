# Polynomial-loss experiments

This directory contains WDRO experiments with polynomial loss functions. Both
experiment scripts follow the shared CLI convention from `../experiment_common.jl` — see
the root [`README`](../../README.md) for the general command shape,
`Distributed`-worker recipe, and full `[experiment]` TOML knob reference.

## Portfolio (`portfolio/`)

Portfolio management with a convex univariate polynomial cost of the portfolio
return. See the
header of `portfolio/portfolio.jl` for the exact parameterization.

Run it with:

```
julia -p 4 --project=MoWDRO.jl/example \
      MoWDRO.jl/example/polynomial_loss/portfolio/portfolio.jl \
      MoWDRO.jl/example/polynomial_loss/portfolio/portfolio.toml \
      result_portfolio.csv
```

The bundled `portfolio.toml` uses `"Wasserstein order" = 4` and
`"baseline method" = "noncvx"` to compare MoWDRO against a nonconvex-QCQP
baseline on the same instance.

## Regression (`regression/`)

Polynomial regression over one of three support sets for the covariate `z`
(`"full-space"`, `"orthant"`, or `"box"`, selected via `[problem]."support set"`).
Two loss families are supported via `[problem]."regression type"`:

* `"mean"` — squared loss `(v − ⟨x, monomials(z)⟩)²`.
* `"quantile"` — pinball loss at level `τ = "quantile level"`.

See the header of `regression/regression.jl` for the exact formulation and the
covariate sampling schemes for each support set.

Run the MoWDRO-only configuration with:

```
julia -p 4 --project=MoWDRO.jl/example \
      MoWDRO.jl/example/polynomial_loss/regression/regression.jl \
      MoWDRO.jl/example/polynomial_loss/regression/regression.toml \
      result_regression.csv
```

TOML variants under `regression/`:

| Config | Baselines | Notes |
| --- | --- | --- |
| `regression.toml` | `"none"` (MoWDRO only) | 10 replications; edit `"regression type"` (and `"quantile level"` if applicable) to switch between squared and pinball loss. |
| `regression_w_noncvx.toml` | `"noncvx"` | Nonconvex-QCQP baseline comparison. `"time limit" = 600` seconds and `"nonconvex baseline bound" = 1.0e4` tighten the support set `Ξ` used inside the nonconvex-QCQP baseline. |

## Output columns

The CSVs share the MoWDRO schema documented in the [linear-recourse
README](../linear_recourse/README.md#output-columns). In addition, quantile
regression runs emit a `QUANTILE_LEVEL` column carrying `τ` on every row, and
`NCVX_*` columns are appended when `"baseline method"` is `"noncvx"` or
`"all"`. All of these are consumed automatically by `../summarize_results.py`
— quantile runs get relabeled per-radius plots without any extra CLI flags.
