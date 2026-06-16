# Shared parsing helpers for the experiment driver scripts under example/.
# Each driver `include`s this file once and then calls
#   CONFIG_PATH = resolve_config_path(@__FILE__)
#   CONFIG      = TOML.parsefile(CONFIG_PATH)
#   ...
#   TRAIN_SIZES = parse_train_sizes(EXP_CFG)
#   WASS_RADII  = parse_wass_radii(EXP_CFG)
#   OUTPUT_FILE = resolve_output_file("result_<name>.csv")

using TOML
using Random

# Resolve the TOML config path: explicit ARGS[1] wins; otherwise look for
# a sibling TOML with the same base name as the script. Pass `@__FILE__`
# from the caller so the error message and sibling lookup point at the
# driver script rather than this helper file.
function resolve_config_path(script_file::AbstractString)
    if length(ARGS) >= 1
        return ARGS[1]
    end
    sibling = joinpath(dirname(script_file),
                       splitext(basename(script_file))[1] * ".toml")
    isfile(sibling) || error(
        "no config supplied and no default sibling TOML at $sibling; " *
        "usage: julia $script_file [<config.toml>]"
    )
    return sibling
end

# Resolve the output CSV path: explicit ARGS[2] override, else the given
# default name placed in the directory from which `julia` was invoked.
function resolve_output_file(default_name::AbstractString)
    if length(ARGS) >= 2
        return ARGS[2]
    end
    return joinpath(pwd(), default_name)
end

# Build a `Vector{T}` from an explicit list under `list_key` and/or a
# vector of {start, stop, step} tables under `sweeps_key`. Internal
# helper shared by `parse_train_sizes` and `parse_wass_radii`.
function _expand_sweeps(::Type{T}, exp_cfg::AbstractDict,
                        list_key::AbstractString,
                        sweeps_key::AbstractString) where {T}
    out = T[]
    if haskey(exp_cfg, list_key)
        append!(out, T.(exp_cfg[list_key]))
    end
    if haskey(exp_cfg, sweeps_key)
        for sw in exp_cfg[sweeps_key]
            append!(out, collect(T(sw["start"]):T(sw["step"]):T(sw["stop"])))
        end
    end
    return out
end

# Parse the training-sample-size sweep from the [experiment] table.
# Accepts any combination of an explicit list under
# `"training sample sizes"` and `{start, stop, step}` sweeps under
# `"training sample sweeps"`; both contribute (concatenated in order).
parse_train_sizes(exp_cfg::AbstractDict) =
    _expand_sweeps(Int, exp_cfg, "training sample sizes", "training sample sweeps")

# Parse the Wasserstein-radii sweep from the [experiment] table.
# Accepts any combination of an explicit list under `"Wasserstein radii"`
# and `{start, stop, step}` sweeps under `"Wasserstein radii sweeps"`.
parse_wass_radii(exp_cfg::AbstractDict) =
    _expand_sweeps(Float64, exp_cfg, "Wasserstein radii", "Wasserstein radii sweeps")

"""
    apply_random_seed!(exp_cfg) -> Union{Int,Nothing}

Honour the optional `"random seed"` key under `[experiment]`. Accepted
values are:

* an integer — passed through to `Random.seed!`; returned to the caller
  so the script can log which seed was applied.
* the string `"none"` (case-insensitive) — interpreted as "no seeding";
  returns `nothing`.
* absent — same as `"none"`.

Anything else is an error.
"""
function apply_random_seed!(exp_cfg::AbstractDict)
    haskey(exp_cfg, "random seed") || return nothing
    v = exp_cfg["random seed"]
    if v isa AbstractString
        lowercase(v) == "none" && return nothing
        error("[experiment] \"random seed\" must be an integer or the string \"none\"; got $(repr(v))")
    elseif v isa Integer
        seed = Int(v)
        Random.seed!(seed)
        return seed
    else
        error("[experiment] \"random seed\" must be an integer or the string \"none\"; got $(repr(v))")
    end
end

"""
    parse_num_reps(exp_cfg) -> Int

Honour the optional `"number of replications"` key under `[experiment]`.
Accepts a positive integer; defaults to 1 when absent. Errors on any other
value.
"""
function parse_num_reps(exp_cfg::AbstractDict)
    haskey(exp_cfg, "number of replications") || return 1
    v = exp_cfg["number of replications"]
    (v isa Integer && v >= 1) || error(
        "[experiment] \"number of replications\" must be a positive integer; got $(repr(v))")
    return Int(v)
end
