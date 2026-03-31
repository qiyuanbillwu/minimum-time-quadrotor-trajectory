module MinTimeQuadTraj

using ForwardDiff
using Ipopt
using MathOptInterface
const MOI = MathOptInterface;

include("setup.jl")

export ProblemMOI, primal_bounds, solve

end