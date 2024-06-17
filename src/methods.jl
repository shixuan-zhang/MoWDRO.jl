## Basic methods for data types conversion and mathematical functions

# build a linear cut from the current state variable and solution information
function build_linear_cut(
        vec_state::Vector{Float64},
        vec_coeff::Vector{Float64},
        val_const::Float64
    )
    return LinearCut(vec_coeff,
                     val_const-vec_coeff'*vec_state)
end
