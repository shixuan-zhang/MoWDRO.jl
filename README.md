# MoWDRO.jl

Implementation of **Mo**ment relaxations for data-driven **W**asserstein **D**istributionally **R**obust **O**ptimization problems.


## Running experiments
Before official release, the experiments on the two-stage production problem can be run through the following command (assuming the working directory contains `MoWDRO.jl`):
```
julia --project=MoWDRO.jl/. MoWDRO.jl/example/linear_recourse/production.jl [output_file]
```
where `[output_file]` is an optional argument for specifying the name of the output file (default to `../result_production_20_20.csv`).
Note that while the module `MoWDRO.jl` only relies on open-source implementations, the experiments are conducted based on commercial solvers `Mosek` and `Gurobi` for better numerical stability.

## Visualizing the results
To help visualizing the results, one can use the helper `Python` script `MoWDRO.jl/example/plot_performance.py`, for example through
```
python MoWDRO.jl/example/plot_performance.py [result_file] [output_dir]
```
where `[result_file]` is the name of the `csv` file containing the experiment results produced as above.
This will produce `TikZ/gnuplot` TeX files in the output directory `[output_dir]`.
