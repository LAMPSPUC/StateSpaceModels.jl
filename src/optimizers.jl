"""
    Optimizer

An Optim.jl wrapper to make the choice of the optimizer straightforward in StateSpaceModels.jl
Users can choose among all suitable Optimizers in Optim.jl using very similar syntax.

# Example
```@jldoctest
julia> using Optim

# use a semicolon to avoid displaying the big log
julia> opt = Optimizer(Optim.LBFGS(), Optim.Options(show_trace = true));
```
"""
mutable struct Optimizer
    method::Optim.AbstractOptimizer
    options::Optim.Options
end

function Optimizer(
    method::Optim.AbstractOptimizer;
    options=Optim.Options(; f_tol=1e-6, g_tol=1e-6, iterations=10^5, show_trace=false),
)
    return Optimizer(method, options)
end