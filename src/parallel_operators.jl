using ThreadsX

Base.getindex(L::AddedOperator, i...) = ThreadsX.sum(op -> op[i...], getops(L))
(*)(          L::AddedOperator, u   ) = ThreadsX.sum(op -> iszero(op) ? zero(u) : op * u, getops(ops))

function LinearAlgebra.mul!(v, L::AddedOperator, u)
    mul!(v, L |> getops |> first, u)
    Threads.@threads for op in getops(L)[2:end]
        iszero(op) && continue
        mul!(v, op, u, true, true)
    end
    v
end

function LinearAlgebra.mul!(v, L::AddedOperator, u, α, β)
    lmul!(β, v)
    Threads.@threads for op in L.ops
        iszero(op) && continue
        mul!(v, op, u, α, true)
    end
    v
end
