export Optimizer

"""
"""
mutable struct Optimizer
    method::Optim.AbstractOptimizer
    options::Optim.Options
end

function Optimizer(method::Optim.AbstractOptimizer;
                   options = Optim.Options(f_tol = 1e-6,
                                           g_tol = 1e-6,
                                           iterations = 10^5,
                                           show_trace = false))
    
    return Optimizer(method, options)
end