struct SparseDiagonallyDominantBridge{T,F,G} <: MOI.Bridges.Constraint.AbstractBridge
    nonzero::Union{Vector{Int}, Nothing}

    # |Qij| variables
    abs_vars::Vector{MOI.VariableIndex}
    # |Qij| ≥ +Qij
    abs_plus::Vector{
        MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},MOI.GreaterThan{T}},
    }
    # |Qij| ≥ -Qij
    abs_minus::Vector{
        MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},MOI.GreaterThan{T}},
    }
    # inequalities Qjj ≥ sum_{i ≠ j} |Qij|
    dominance::Vector{MOI.ConstraintIndex{F,MOI.GreaterThan{T}}}
end

function matrix_index(k)
    # Vectorized index to matrix index computation from
    # https://jump.dev/MathOptInterface.jl/stable/reference/standard_form/#MathOptInterface.AbstractSymmetricMatrixSetTriangle
    j = div(1 + isqrt(8k - 7), 2)
    i = k - div((j - 1) * j, 2)

    return j, i
end

function MOI.Bridges.Constraint.bridge_constraint(
    ::Type{SparseDiagonallyDominantBridge{T,F,G}},
    model::MOI.ModelLike,
    f::MOI.AbstractVectorFunction,
    s::SOS.SparseDiagonallyDominantCone,
) where {T,F,G}
    @assert MOI.output_dimension(f) == MOI.dimension(s)

    n = s.side_dimension
    g = F[zero(F) for i in 1:n]
    
    fs = MOI.Utilities.eachscalar(f)
    isdiagonal((j, i)) = j == i

    if isnothing(s.nonzero)
        matrix_indices = ((j, i) for j in 1:side_dim, i in 1:(j-1))
        num_off_diag = MOI.dimension(s) - n
    else
        matrix_indices = map(matrix_index, s.nonzero)
        num_off_diag = MOI.dimension(s) - count(isdiagonal, matrix_indices)
    end

    CI = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},MOI.GreaterThan{T}}
    abs_vars = Vector{MOI.VariableIndex}(undef, num_off_diag)
    abs_plus = Vector{CI}(undef, num_off_diag)
    abs_minus = Vector{CI}(undef, num_off_diag)
    koff = 0

    for ((j, i), f) in zip(matrix_indices, fs)
        if isdiagonal((j, i))
            # Qjj ≥ sum_{i ≠ j} |Qij|
            MOI.Utilities.operate!(+, T, g[j], f)
        else
            koff += 1
            # abs ≥ |Qij|
            abs_vars[koff] = MOI.add_variable(model)
            fabs = abs_vars[koff]
            MOI.Utilities.operate!(-, T, g[j], fabs)
            MOI.Utilities.operate!(-, T, g[i], fabs)
            abs_plus[koff] = MOI.add_constraint(
                model,
                MOI.Utilities.operate(+, T, fabs, f),
                MOI.GreaterThan(0.0),
            )
            abs_minus[koff] = MOI.add_constraint(
                model,
                MOI.Utilities.operate(-, T, fabs, f),
                MOI.GreaterThan(0.0),
            )
        end
    end

    dominance = map(f -> MOI.add_constraint(model, f, MOI.GreaterThan(0.0)), g)
    return SparseDiagonallyDominantBridge{T,F,G}(
        s.nonzero,
        abs_vars,
        abs_plus,
        abs_minus,
        dominance,
    )
end

function MOI.supports_constraint(
    ::Type{<:SparseDiagonallyDominantBridge},
    ::Type{<:MOI.AbstractVectorFunction},
    ::Type{<:SOS.SparseDiagonallyDominantCone},
)
    return true
end
function MOI.Bridges.added_constrained_variable_types(
    ::Type{<:SparseDiagonallyDominantBridge},
)
    return Tuple{DataType}[]
end
function MOI.Bridges.added_constraint_types(
    ::Type{<:SparseDiagonallyDominantBridge{T,F}},
) where {T,F}
    added = [(F, MOI.GreaterThan{T})]
    if F != MOI.ScalarAffineFunction{T}
        push!(added, (MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}))
    end
    return added
end
function MOI.Bridges.Constraint.concrete_bridge_type(
    ::Type{<:SparseDiagonallyDominantBridge{T}},
    G::Type{<:MOI.AbstractVectorFunction},
    ::Type{SOS.SparseDiagonallyDominantCone},
) where {T}
    S = MOI.Utilities.scalar_type(G)
    F = MOI.Utilities.promote_operation(-, T, S, MOI.VariableIndex)
    return SparseDiagonallyDominantBridge{T,F,G}
end

# Attributes, Bridge acting as an model
function MOI.get(bridge::SparseDiagonallyDominantBridge, ::MOI.NumberOfVariables)
    return length(bridge.abs_vars)
end
function MOI.get(bridge::SparseDiagonallyDominantBridge, ::MOI.ListOfVariableIndices)
    return bridge.abs_vars
end
function MOI.get(
    bridge::SparseDiagonallyDominantBridge{T,MOI.ScalarAffineFunction{T}},
    ::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{T},MOI.GreaterThan{T}},
) where {T}
    return length(bridge.abs_plus) +
           length(bridge.abs_minus) +
           length(bridge.dominance)
end
function MOI.get(
    bridge::SparseDiagonallyDominantBridge{T},
    ::MOI.NumberOfConstraints{MOI.ScalarAffineFunction{T},MOI.GreaterThan{T}},
) where {T}
    return length(bridge.abs_plus) + length(bridge.abs_minus)
end
function MOI.get(
    bridge::SparseDiagonallyDominantBridge{T,F},
    ::MOI.NumberOfConstraints{F,MOI.GreaterThan{T}},
) where {T,F}
    return length(bridge.dominance)
end
function MOI.get(
    bridge::SparseDiagonallyDominantBridge{T,MOI.ScalarAffineFunction{T}},
    ::MOI.ListOfConstraintIndices{
        MOI.ScalarAffineFunction{T},
        MOI.GreaterThan{T},
    },
) where {T}
    return vcat(bridge.abs_plus, bridge.abs_minus, bridge.dominance)
end
function MOI.get(
    bridge::SparseDiagonallyDominantBridge{T},
    ::MOI.ListOfConstraintIndices{
        MOI.ScalarAffineFunction{T},
        MOI.GreaterThan{T},
    },
) where {T}
    return vcat(bridge.abs_plus, bridge.abs_minus)
end
function MOI.get(
    bridge::SparseDiagonallyDominantBridge{T,F},
    ::MOI.ListOfConstraintIndices{F,MOI.GreaterThan{T}},
) where {T,F}
    return bridge.dominance
end

# Indices
function MOI.delete(model::MOI.ModelLike, bridge::SparseDiagonallyDominantBridge)
    for ci in bridge.dominance
        MOI.delete(model, ci)
    end
    for ci in bridge.abs_plus
        MOI.delete(model, ci)
    end
    for ci in bridge.abs_minus
        MOI.delete(model, ci)
    end
    for vi in bridge.abs_vars
        MOI.delete(model, vi)
    end
end

# Attributes, Bridge acting as a constraint
function MOI.get(
    ::MOI.ModelLike,
    ::MOI.ConstraintSet,
    bridge::SparseDiagonallyDominantBridge,
)
    return SOS.SparseDiagonallyDominantCone(length(bridge.dominance), bridge.nonzero)
end
function MOI.get(
    model::MOI.ModelLike,
    attr::MOI.ConstraintFunction,
    bridge::SparseDiagonallyDominantBridge{T,F,G},
) where {T,F,G}
    set = MOI.get(model, MOI.ConstraintSet(), bridge)
    H = MOI.Utilities.scalar_type(G)
    g = Vector{H}(undef, MOI.dimension(set))
    koff = 0

    if isnothing(set.nonzero)
        vectorized_indices = 1:MOI.dimension(set)
    else
        vectorized_indices = set.nonzero
    end

    for k in vectorized_indices
        j, i = matrix_index(k)

        if j == i
            func = MOI.get(model, attr, bridge.dominance[j])
        else
            koff += 1
            func = MOI.get(model, attr, bridge.abs_plus[koff])
        end

        g[k] = MOI.Utilities.convert_approx(
            H,
            MOI.Utilities.remove_variable(func, bridge.abs_vars),
        )
    end
    return MOI.Utilities.vectorize(g)
end

# TODO ConstraintPrimal

function MOI.get(
    model::MOI.ModelLike,
    attr::MOI.ConstraintDual,
    bridge::SparseDiagonallyDominantBridge{T},
) where {T}
    dominance_dual = MOI.get(model, attr, bridge.dominance)
    side_dim = length(dominance_dual)
    dim = MOI.dimension(SOS.SparseDiagonallyDominantCone(side_dim, bridge.nonzero))
    dual = Array{T}(undef, dim)
    k = 0
    for j in 1:side_dim
        for i in 1:(j-1)
            k += 1
            # Need to divide by 2 because of the custom scalar product for this
            # cone
            dual[k] = (-dominance_dual[i] - dominance_dual[j]) / 2
        end
        k += 1
        dual[k] = dominance_dual[j]
    end
    @assert k == dim
    return dual
end
