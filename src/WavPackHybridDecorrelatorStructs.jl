# ============================================================
# Weight application and update (WavPack-accurate)
# ============================================================

@inline function apply_weight(weight::Int32, sample::Int32)
    return Int32((Int64(weight) * Int64(sample) + 512) >> 10)
end

@inline function update_weight(weight::Int32, delta::Int32, source::Int32, result::Int32)
    if source != 0 && result != 0
        s = (source ⊻ result) >> 31
        weight = (delta ⊻ s) + (weight - s)
    end
    return weight
end

@inline function update_weight_clip(weight::Int32, delta::Int32, source::Int32, result::Int32)
    if source != 0 && result != 0
        s = (source ⊻ result) >> 31
        weight = (weight ⊻ s) + (delta - s)
        weight = weight > 1024 ? 1024 : weight
        weight = (weight ⊻ s) - s
    end
    return weight
end

# ============================================================
# Pass memory types
# ============================================================

struct IntraPassMemoryGeneric{N,T}
    delta::Int32
end

struct IntraPassMemorySpecial{term,T}
    delta::Int32
end

struct IntraPassMemoryCrossChannel{term}
    delta::Int32
end

# ============================================================
# Predictor state types
# ============================================================

struct IntraPassState
    weight::Int32
    idx::Int
end

struct CrossPassState
    weight::Int32
end

# ============================================================
# Intra-channel decorrelation (generic terms 1–8)
# ============================================================

@inline function process_pass!(
    mem::IntraPassMemoryGeneric{N,T},
    st::IntraPassState,
    buf::MMatrix{2,TOTAL,T},
    offset::Int,
    s::Stereo{T},
) where {N,T,TOTAL}

    L = s.l
    R = s.r

    bi = offset + st.idx
    dL = buf[1, bi]
    dR = buf[2, bi]

    predL = (st.weight * dL) >>> 10
    predR = (st.weight * dR) >>> 10

    resL = L - predL
    resR = R - predR

    new_weight = st.weight + (resL > 0 ? mem.delta : (resL < 0 ? -mem.delta : 0))
    new_idx    = st.idx == N ? 1 : st.idx + 1

    buf[1, bi] = L
    buf[2, bi] = R

    return Stereo{T}(resL, resR), IntraPassState(new_weight, new_idx)
end

# ============================================================
# Intra-channel decorrelation (special terms 17 & 18)
# ============================================================

@inline function process_pass!(
    mem::IntraPassMemorySpecial{term,T},
    st::IntraPassState,
    buf::MMatrix{2,TOTAL,T},
    offset::Int,
    s::Stereo{T},
) where {term,T,TOTAL}

    L = s.l
    R = s.r

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
    else
        error("Invalid special term")
    end

    predL = (st.weight * samL) >>> 10
    predR = (st.weight * samR) >>> 10

    resL = L - predL
    resR = R - predR

    new_weight = st.weight + (resL > 0 ? mem.delta : (resL < 0 ? -mem.delta : 0))

    buf[1, i1] = xL0
    buf[2, i1] = xR0
    buf[1, i0] = L
    buf[2, i0] = R

    new_idx = st.idx == 2 ? 1 : st.idx + 1

    return Stereo{T}(resL, resR), IntraPassState(new_weight, new_idx)
end

# ============================================================
# Cross-channel decorrelation (-1, -2, -3)
# ============================================================

@inline function process_pass!(
    mem::IntraPassMemoryCrossChannel{term},
    st::CrossPassState,
    s::Stereo{T},
) where {T,term}

    L = s.l; R = s.r

    if term == -1
        pred = (st.weight * L) >>> 10
        resR = R - pred
        resL = L
        new_weight = st.weight + (resR > 0 ? mem.delta : (resR < 0 ? -mem.delta : 0))
        return Stereo{T}(resL, resR), CrossPassState(new_weight)

    elseif term == -2
        pred = (st.weight * R) >>> 10
        resL = L - pred
        resR = R
        new_weight = st.weight + (resL > 0 ? mem.delta : (resL < 0 ? -mem.delta : 0))
        return Stereo{T}(resL, resR), CrossPassState(new_weight)

    elseif term == -3
        predL = (st.weight * R) >>> 10
        predR = (st.weight * L) >>> 10
        resL = L - predL
        resR = R - predR
        err = resL + resR
        new_weight = st.weight + (err > 0 ? mem.delta : (err < 0 ? -mem.delta : 0))
        return Stereo{T}(resL, resR), CrossPassState(new_weight)
    end
end

@inline process_pass!(
    mem::IntraPassMemoryCrossChannel{term},
    st::CrossPassState,
    buf::MMatrix{2,TOTAL,T},
    offset::Int,
    s::Stereo{T},
) where {T,term,TOTAL} = process_pass!(mem, st, s)

# ============================================================
# Memory + state constructors
# ============================================================

function make_memories(::Type{T}, terms, deltas) where {T}
    mems = ()
    for (term, delta) in zip(terms, deltas)
        if 1 ≤ term ≤ 8
            mems = (mems..., IntraPassMemoryGeneric{term,T}(delta))
        elseif term == 17
            mems = (mems..., IntraPassMemorySpecial{17,T}(delta))
        elseif term == 18
            mems = (mems..., IntraPassMemorySpecial{18,T}(delta))
        else
            mems = (mems..., IntraPassMemoryCrossChannel{term}(delta))
        end
    end
    return mems
end

function make_states(memories; init_weight=0)
    states = ()
    for mem in memories
        if mem isa IntraPassMemoryGeneric || mem isa IntraPassMemorySpecial
            states = (states..., IntraPassState(init_weight, 1))
        else
            states = (states..., CrossPassState(init_weight))
        end
    end
    return states
end

# ============================================================
# Compute offsets for buffer
# ============================================================

function compute_offsets(memories)
    offsets = ()
    offset = 0
    for mem in memories
        if mem isa IntraPassMemoryGeneric{N,T} where {N,T}
            offsets = (offsets..., offset)
            offset += N
        elseif mem isa IntraPassMemorySpecial{term,T} where {term,T}
            offsets = (offsets..., offset)
            offset += 2
        else
            offsets = (offsets..., 0)
        end
    end
    return offsets, max(offset, 1)
end

# ============================================================
# Generated decorrelation chain
# ============================================================

@generated function process_chain!(
    memories::MT,
    states::ST,
    offsets::OT,
    buf::MMatrix{2,TOTAL,T},
    s::Stereo{T},
) where {N,MT<:NTuple{N,Any},ST<:NTuple{N,Any},OT<:NTuple{N,Any},TOTAL,T}

    steps = [:(s, st_$i = process_pass!(memories[$i], states[$i], buf, offsets[$i], s)) for i in 1:N]
    new_states = Expr(:tuple, [Symbol("st_$i") for i in 1:N]...)

    quote
        $(Expr(:block, steps...))
        return s, $new_states
    end
end

# ============================================================
# Hybrid shaping (WavPack-accurate)
# ============================================================

@inline function hybrid_shape_sample!(
    sample::Int32,
    err::Int32,
    shaping_acc::Int32,
    shaping_delta::Int32,
    shaping_array::Union{Nothing,Vector{Int16}},
    idx::Int,
    flags::UInt32
)
    HYBRID_SHAPE = 0x00000008
    NEW_SHAPING  = 0x00000010

    shaping_weight::Int32

    if shaping_array !== nothing
        shaping_weight = shaping_array[idx]
    else
        shaping_acc += shaping_delta
        shaping_weight = shaping_acc >> 16
    end

    temp = -apply_weight(shaping_weight, err)

    if (flags & NEW_SHAPING != 0) && shaping_weight < 0 && temp != 0
        if temp == err
            temp += temp < 0 ? 1 : -1
        end
        err = -sample
        sample += temp
    else
        sample += temp
        err = -sample
    end

    return sample, err, shaping_acc
end

# ============================================================
# Quantization (WavPack-accurate)
# ============================================================

@inline function quantize_residual(res::Int32, q::Int32)
    half_q = q >>> 1
    qv = (res + half_q) ÷ q
    dq = qv * q
    corr = res - dq
    return qv, corr
end

# ============================================================
# Full hybrid block (shaping + decorrelation + quantization)
# ============================================================

function hybrid_block(
    data::AbstractVector{Stereo{Int32}},
    memories;
    init_weight::Int32 = 0,
    qL::Int32,
    qR::Int32,
    shaping_delta_L::Int32,
    shaping_delta_R::Int32,
    flags::UInt32,
    shaping_array_L::Union{Nothing,Vector{Int16}} = nothing,
    shaping_array_R::Union{Nothing,Vector{Int16}} = nothing
)

    states = make_states(memories; init_weight)
    offsets, TOTAL = compute_offsets(memories)
    buf = MMatrix{2,TOTAL,Int32}(zeros(Int32, 2, TOTAL))

    n = length(data)
    quantized  = Vector{Stereo{Int32}}(undef, n)
    correction = Vector{Stereo{Int32}}(undef, n)

    errL = Int32(0)
    errR = Int32(0)
    shaping_acc_L = Int32(0)
    shaping_acc_R = Int32(0)

    for i in 1:n
        L = data[i].l
        R = data[i].r

        # 1. Pre-decorrelation shaping
        L, errL, shaping_acc_L = hybrid_shape_sample!(
            L, errL, shaping_acc_L, shaping_delta_L,
            shaping_array_L, i, flags
        )

        R, errR, shaping_acc_R = hybrid_shape_sample!(
            R, errR, shaping_acc_R, shaping_delta_R,
            shaping_array_R, i, flags
        )

        # 2. Decorrelate
        res, states = process_chain!(memories, states, offsets, buf, Stereo{Int32}(L, R))

        # 3. Quantize residuals
        qvL, corrL = quantize_residual(res.l, qL)
        qvR, corrR = quantize_residual(res.r, qR)

        quantized[i]  = Stereo{Int32}(qvL, qvR)
        correction[i] = Stereo{Int32}(corrL, corrR)

        # 4. Post-quantization error feedback
        errL += qvL * qL
        errR += qvR * qR
    end

    return quantized, correction, states
end
