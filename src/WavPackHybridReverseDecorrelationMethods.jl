# ============================================================
# Inverse decorrelation passes
# ============================================================

@inline function inverse_pass!(
    mem::IntraPassMemoryGeneric{N,T},
    st::IntraPassState,
    buf::MMatrix{2,TOTAL,T},
    offset::Int,
    res::Stereo{T},
) where {N,T,TOTAL}

    bi = offset + st.idx
    dL = buf[1, bi]
    dR = buf[2, bi]

    predL = (st.weight * dL) >>> 10
    predR = (st.weight * dR) >>> 10

    L = res.l + predL
    R = res.r + predR

    new_weight = st.weight + (res.l > 0 ? mem.delta : (res.l < 0 ? -mem.delta : 0))
    new_idx    = st.idx == N ? 1 : st.idx + 1

    buf[1, bi] = L
    buf[2, bi] = R

    return Stereo{T}(L, R), IntraPassState(new_weight, new_idx)
end

@inline function inverse_pass!(
    mem::IntraPassMemorySpecial{term,T},
    st::IntraPassState,
    buf::MMatrix{2,TOTAL,T},
    offset::Int,
    res::Stereo{T},
) where {term,T,TOTAL}

    i0 = offset + st.idx
    i1 = offset + (st.idx == 1 ? 2 : st.idx - 1)

    xL0 = buf[1, i0]; xL1 = buf[1, i1]
    xR0 = buf[2, i0]; xR1 = buf[2, i1]

    if term == 17
        samL = 2*xL0 - xL1
        samR = 2*xR0 - xR1
    elseif term == 18
        samL = xL0 + ((xL0 - xL1) >>> 1)
        samR = xR0 + ((xR0 - xR1) >>> 1)
    end

    predL = (st.weight * samL) >>> 10
    predR = (st.weight * samR) >>> 10

    L = res.l + predL
    R = res.r + predR

    new_weight = st.weight + (res.l > 0 ? mem.delta : (res.l < 0 ? -mem.delta : 0))
    new_idx = st.idx == 2 ? 1 : st.idx + 1

    buf[1, i1] = xL0
    buf[2, i1] = xR0
    buf[1, i0] = L
    buf[2, i0] = R

    return Stereo{T}(L, R), IntraPassState(new_weight, new_idx)
end

@inline function inverse_pass!(
    mem::IntraPassMemoryCrossChannel{term},
    st::CrossPassState,
    res::Stereo{T},
) where {T,term}

    if term == -1
        pred = (st.weight * res.l) >>> 10
        R = res.r + pred
        L = res.l
        new_weight = st.weight + (res.r > 0 ? mem.delta : (res.r < 0 ? -mem.delta : 0))
        return Stereo{T}(L, R), CrossPassState(new_weight)

    elseif term == -2
        pred = (st.weight * res.r) >>> 10
        L = res.l + pred
        R = res.r
        new_weight = st.weight + (res.l > 0 ? mem.delta : (res.l < 0 ? -mem.delta : 0))
        return Stereo{T}(L, R), CrossPassState(new_weight)

    elseif term == -3
        predL = (st.weight * res.r) >>> 10
        predR = (st.weight * res.l) >>> 10
        L = res.l + predL
        R = res.r + predR
        err = res.l + res.r
        new_weight = st.weight + (err > 0 ? mem.delta : (err < 0 ? -mem.delta : 0))
        return Stereo{T}(L, R), CrossPassState(new_weight)
    end
end

@inline inverse_pass!(
    mem::IntraPassMemoryCrossChannel{term},
    st::CrossPassState,
    buf::MMatrix{2,TOTAL,T},
    offset::Int,
    res::Stereo{T},
) where {T,term,TOTAL} = inverse_pass!(mem, st, res)

# ============================================================
# Generated inverse decorrelation chain
# ============================================================

@generated function inverse_chain!(
    memories::MT,
    states::ST,
    offsets::OT,
    buf::MMatrix{2,TOTAL,T},
    res::Stereo{T},
) where {N,MT<:NTuple{N,Any},ST<:NTuple{N,Any},OT<:NTuple{N,Any},TOTAL,T}

    steps = [:(res, st_$i = inverse_pass!(memories[$i], states[$i], buf, offsets[$i], res)) for i in N:-1:1]
    new_states = Expr(:tuple, [Symbol("st_$i") for i in 1:N]...)

    quote
        $(Expr(:block, steps...))
        return res, $new_states
    end
end

# ============================================================
# Hybrid decoder
# ============================================================

function hybrid_decode(
    quantized::Vector{Stereo{Int32}},
    correction::Vector{Stereo{Int32}},
    memories,
    qL::Int32,
    qR::Int32,
    init_weight::Int32 = 0
)

    states = make_states(memories; init_weight)
    offsets, TOTAL = compute_offsets(memories)
    buf = MMatrix{2,TOTAL,Int32}(zeros(Int32, 2, TOTAL))

    n = length(quantized)
    pcm = Vector{Stereo{Int32}}(undef, n)

    for i in 1:n
        qvL = quantized[i].l
        qvR = quantized[i].r

        corrL = correction[i].l
        corrR = correction[i].r

        # 1. Reconstruct residuals
        resL = qvL * qL + corrL
        resR = qvR * qR + corrR

        # 2. Inverse decorrelation
        pcm[i], states = inverse_chain!(memories, states, offsets, buf, Stereo{Int32}(resL, resR))
    end

    return pcm
end
