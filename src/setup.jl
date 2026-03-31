struct ProblemMOI{F,G} <: MOI.AbstractNLPEvaluator
    n_nlp::Int
    m_nlp::Int
    Nt::Int
    Nx::Int
    Nu::Int
    obj_fun::F
    con_fun!::G
    idx_ineq
    obj_grad::Bool
    con_jac::Bool
    sparsity_jac
    sparsity_hess
    primal_bounds
    constraint_bounds
    hessian_lagrangian::Bool
end

function ProblemMOI(n_nlp,m_nlp, Nt, Nx, Nu, obj_fun, con_fun!;
    idx_ineq=(1:0),
    obj_grad=true,
    con_jac=true,
    sparsity_jac=sparsity_jacobian(n_nlp,m_nlp),
    sparsity_hess=sparsity_hessian(n_nlp,m_nlp),
    primal_bounds=primal_bounds(n_nlp, Nt, Nx, Nu),
    constraint_bounds=constraint_bounds(m_nlp,idx_ineq=idx_ineq),
    hessian_lagrangian=false)

    ProblemMOI(
        n_nlp,
        m_nlp,
        Nt,
        Nx,
        Nu,
        obj_fun,
        con_fun!,
        idx_ineq,
        obj_grad,
        con_jac,
        sparsity_jac,
        sparsity_hess,
        primal_bounds,
        constraint_bounds,
        hessian_lagrangian)
end

function primal_bounds(n, Nt, Nx, Nu; T_min=1.0e-6, g=9.81, u1_min=1/g, u1_max=20/g, u2_min=-1.0, u2_max=1.0)
    x_l = -Inf*ones(n)
    x_u = Inf*ones(n)
    x_l[1] = T_min
    for k = 1:Nt
        idx_u1 = 1 + (k-1)*(Nx+Nu) + (Nx+1)
        idx_u2 = idx_u1 + 1
        x_l[idx_u1] = u1_min
        x_u[idx_u1] = u1_max
        x_l[idx_u2] = u2_min
        x_u[idx_u2] = u2_max
    end
    return x_l, x_u
end

function constraint_bounds(m; idx_ineq=(1:0))
    c_l = zeros(m)
    c_l[idx_ineq] .= -Inf

    c_u = zeros(m)
    return c_l, c_u
end

function row_col!(row,col,r,c)
    for cc in c
        for rr in r
            push!(row,rr)
            push!(col,cc)
        end
    end
    return row, col
end

function sparsity_jacobian(n,m)

    row = Int[]
    col = Int[]

    r = 1:m
    c = 1:n

    row_col!(row,col,r,c)

    return [(row[i], col[i]) for i in 1:length(row)]
end

function sparsity_hessian(n,m)

    row = Int[]
    col = Int[]

    r = 1:m
    c = 1:n

    row_col!(row,col,r,c)

    return [(row[i], col[i]) for i in 1:length(row)]
end

function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    prob.obj_fun(x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    ForwardDiff.gradient!(grad_f,prob.obj_fun,x)
    return nothing
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,g,x)
    prob.con_fun!(g,x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    ForwardDiff.jacobian!(reshape(jac,prob.m_nlp,prob.n_nlp), prob.con_fun!, zeros(prob.m_nlp), x)
    return nothing
end

function MOI.features_available(prob::MOI.AbstractNLPEvaluator)
    return [:Grad, :Jac]
end

MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = prob.sparsity_jac

function solve(x0,prob::MOI.AbstractNLPEvaluator;
        tol=1.0e-4,c_tol=1.0e-4,max_iter=1000)
    x_l, x_u = prob.primal_bounds
    c_l, c_u = prob.constraint_bounds

    nlp_bounds = MOI.NLPBoundsPair.(c_l,c_u)
    block_data = MOI.NLPBlockData(nlp_bounds,prob,true)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iter
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol

    x = MOI.add_variables(solver,prob.n_nlp)

    for i = 1:prob.n_nlp
        MOI.add_constraint(solver, x[i], MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, x[i], MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), x)

    return res
end