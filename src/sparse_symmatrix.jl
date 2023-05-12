export SparseSymMatrix

struct SparseSymMatrix{T} <: AbstractMatrix{T}
    Q::Vector{T}
    nonzero::Vector{Int}
    n::Int
end

function SparseSymMatrix{T}(dims::Dims{2}) where T
    dims[1] != dims[2] && error("Expected same dimension for `SparseSymMatrix`, got `$(dims)`.")
    n = dims[1]
    return SparseSymMatrix(Vector{T}(), Vector{Int}(), n)
end
Base.similar(Q::SparseSymMatrix{T}, dims::Tuple{Base.OneTo, Vararg{Base.OneTo}}) where {T} = similar(Matrix{T}, dims)
Base.similar(Q::SparseSymMatrix, T::Type, dims::Tuple{Base.OneTo, Vararg{Base.OneTo}}) = similar(Matrix{T}, dims)
Base.similar(Q::SparseSymMatrix, T::Type, dims::Dims{2}) = similar(SparseSymMatrix{T}, dims)
Base.copy(Q::SparseSymMatrix) = SparseSymMatrix(copy(Q.Q), copy(Q.nonzero), Q.n)
Base.map(f::Function, Q::SparseSymMatrix) = SparseSymMatrix(map(f, Q.Q), copy(Q.nonzero), Q.n)

# i <= j
trimap(i, j) = div(j * (j - 1), 2) + i

# function trimat(::Type{T}, f, n, σ) where {T}
#     Q = _undef_sym(T, n)
#     for j in 1:n
#         for i in 1:j
#             Q[trimap(i, j)] = f(σ[i], σ[j])
#         end
#     end
#     return SymMatrix(Q, n)
# end

Base.size(Q::SparseSymMatrix) = (Q.n, Q.n)

# """
#     square_getindex!(Q::SymMatrix, I)

# Return the `SymMatrix` corresponding to `Q[I, I]`.
# """
# function square_getindex(Q::SymMatrix{T}, I) where T
#     n = length(I)
#     q = _undef_sym(T, n)
#     k = 0
#     for (j, Ij) in enumerate(I)
#         for (i, Ii) in enumerate(I)
#             i > j && break
#             k += 1
#             q[k] = Q[Ii, Ij]
#         end
#     end
#     return SymMatrix(q, n)
# end

# """
#     symmetric_setindex!(Q::SymMatrix, value, i::Integer, j::Integer)

# Set `Q[i, j]` and `Q[j, i]` to the value `value`.
# """
# function symmetric_setindex!(Q::SymMatrix, value, i::Integer, j::Integer)
#     Q.Q[trimap(min(i, j), max(i, j))] = value
# end

function Base.getindex(Q::SparseSymMatrix{T}, i::Integer, j::Integer) where T
    k = trimap(min(i, j), max(i, j))
    idx = findfirst(elem -> elem == k, Q.nonzero)

    if isnothing(idx)
        return zero(T)
    else
        return Q.Q[idx]
    end
end
Base.getindex(Q::SparseSymMatrix, I::Tuple) = Q[I...]
Base.getindex(Q::SparseSymMatrix, I::CartesianIndex) = Q[I.I]