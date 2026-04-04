module WavPackHybridDecorrelators

using StaticArrays
using SamplesCore: Stereo

# ------------------------------------------------------------
# Pass memory types
# ------------------------------------------------------------

struct IntraPassMemoryGeneric{N,T}
    delta::Int
end

struct IntraPassMemorySpecial{N,T}
    delta::Int
end

struct IntraPassMemoryCrossChannel{term}
    delta::Int
end

# ------------------------------------------------------------
# Predictor state types
# ------------------------------------------------------------

struct IntraPassState
    weight::Int
    idx::Int
end

struct CrossPassState
    weight::Int
end

# ------------------------------------------------------------
# Intra-channel decorrelation pass
# ------------------------------------------------------------

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

@inline function process_pass!(
    mem::IntraPassMemorySpecial{N,T},
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

# ------------------------------------------------------------
# Cross-channel decorrelation pass
# ------------------------------------------------------------

@inline function process_pass!(
    mem::IntraPassMemoryCrossChannel{term},
    st::CrossPassState,
    s::Stereo{T},
) where {T,term}

    L = s.l
    R = s.r

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

# unify arity for generated chain
@inline process_pass!(
    mem::IntraPassMemoryCrossChannel{term},
    st::CrossPassState,
    buf::MMatrix{2,TOTAL,T},
    offset::Int,
    s::Stereo{T},
) where {T,term,TOTAL} = process_pass!(mem, st, s)

# ------------------------------------------------------------
# Memory + state constructors
# ------------------------------------------------------------

function make_memories(::Type{T}, terms, deltas) where {T}
    mems = ()
    for (term, delta) in zip(terms, deltas)
        if 1 ≤ term ≤ 8
            mems = (mems..., IntraPassMemoryGeneric{term,T}(delta))
        elseif term == 17
            mems = (mems..., IntraPassMemorySpecial{1,T}(delta))
        elseif term == 18
            mems = (mems..., IntraPassMemorySpecial{2,T}(delta))
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

# ------------------------------------------------------------
# Compute offsets for buffer
# ------------------------------------------------------------

function compute_offsets(memories)
    offsets = ()
    offset = 0
    for mem in memories
        if mem isa IntraPassMemoryGeneric{N,T} where {N,T}
            offsets = (offsets..., offset)
            offset += N
        elseif mem isa IntraPassMemorySpecial{N,T} where {N,T}
            offsets = (offsets..., offset)
            offset += N
        else
            offsets = (offsets..., 0)
        end
    end
    return offsets, max(offset, 1)
end

# ------------------------------------------------------------
# Generated decorrelation chain
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Per-channel hybrid quantizer (integer)
# ------------------------------------------------------------

@inline function quantize_channel(res::T,
                                  err::T,
                                  q::T,
                                  α::T) where {T}

    shaped = res + α * err
    half_q = q >>> 1
    qv = (shaped + half_q) ÷ q
    dq = qv * q
    corr = res - dq
    new_err = shaped - dq

    return qv, corr, new_err
end

# ------------------------------------------------------------
# Fused hybrid block (per-channel shaping + quantization)
# ------------------------------------------------------------

function hybrid_block(data::AbstractVector{Stereo{T}},
                      memories;
                      init_weight::Int=0,
                      qL::T,
                      qR::T,
                      shapeL::T,
                      shapeR::T) where {T}

    states = make_states(memories; init_weight)
    offsets, TOTAL = compute_offsets(memories)
    buf = MMatrix{2,TOTAL,T}(zeros(T, 2, TOTAL))

    n = length(data)
    quantized  = Vector{Stereo{Int}}(undef, n)
    correction = Vector{Stereo{T}}(undef, n)

    errL = zero(T)
    errR = zero(T)

    for i in eachindex(data)
        res, states = process_chain!(memories, states, offsets, buf, data[i])

        ql, corrL, errL = quantize_channel(res.l, errL, qL, shapeL)
        qr, corrR, errR = quantize_channel(res.r, errR, qR, shapeR)

        quantized[i]  = Stereo{Int}(ql, qr)
        correction[i] = Stereo{T}(corrL, corrR)
    end

    return quantized, correction, states
end

end # module
