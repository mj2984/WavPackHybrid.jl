module WavPackHybridDecorrelators

using StaticArrays
using SamplesCore: Stereo

# ------------------------------------------------------------
# Pass memory types (algorithmic only, no layout)
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
# Immutable state types
# ------------------------------------------------------------

struct IntraPassState
    weight::Int
    idx::Int
end

struct CrossPassState
    weight::Int
end

# ------------------------------------------------------------
# Intra-channel processing (single MMatrix buffer + per-pass offset)
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
# Cross-channel processing (no buffer use)
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

# unify arity for generated chain (ignore buf/offset)
@inline process_pass!(
    mem::IntraPassMemoryCrossChannel{term},
    st::CrossPassState,
    buf::MMatrix{2,TOTAL,T},
    offset::Int,
    s::Stereo{T},
) where {T,term,TOTAL} = process_pass!(mem, st, s)

# ------------------------------------------------------------
# Constructors for memories and states
# ------------------------------------------------------------

function make_memories(::Type{T},
                       terms::AbstractVector{Int},
                       deltas::AbstractVector{Int}) where {T}

    mems = ()
    for (term, delta) in zip(terms, deltas)
        if 1 <= term <= 8
            N = term
            mem = IntraPassMemoryGeneric{N,T}(delta)
        elseif term == 17
            N = 1
            mem = IntraPassMemorySpecial{N,T}(delta)
        elseif term == 18
            N = 2
            mem = IntraPassMemorySpecial{N,T}(delta)
        else
            mem = IntraPassMemoryCrossChannel{term}(delta)
        end
        mems = (mems..., mem)
    end
    return mems
end

function make_states(memories; init_weight=0)
    states = ()
    for mem in memories
        if mem isa IntraPassMemoryGeneric || mem isa IntraPassMemorySpecial
            st = IntraPassState(init_weight, 1)
        else
            st = CrossPassState(init_weight)
        end
        states = (states..., st)
    end
    return states
end

# ------------------------------------------------------------
# Compute per-pass offsets + total buffer size
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
            offsets = (offsets..., 0)  # cross-channel: unused
        end
    end
    return offsets, max(offset, 1)
end

# ------------------------------------------------------------
# Generated, fully unrolled chain (with offsets)
# ------------------------------------------------------------

@generated function process_chain!(
    memories::MT,
    states::ST,
    offsets::OT,
    buf::MMatrix{2,TOTAL,T},
    s::Stereo{T},
) where {N,MT<:NTuple{N,Any},ST<:NTuple{N,Any},OT<:NTuple{N,Any},TOTAL,T}

    steps = Vector{Expr}(undef, N)
    for i in 1:N
        steps[i] = :(s, st_$i = process_pass!(memories[$i], states[$i], buf, offsets[$i], s))
    end

    new_states = Expr(:tuple, [Symbol("st_$i") for i in 1:N]...)

    quote
        $(Expr(:block, steps...))
        return s, $new_states
    end
end

# ------------------------------------------------------------
# Lossless-style block processor (in-place residuals)
# ------------------------------------------------------------

function process_block!(data::AbstractVector{Stereo{T}},
                        memories;
                        init_weight=0) where {T}

    states = make_states(memories; init_weight)
    offsets, TOTAL = compute_offsets(memories)
    buf = MMatrix{2,TOTAL,T}(zeros(T, 2, TOTAL))

    for i in eachindex(data)
        s = data[i]
        s, states = process_chain!(memories, states, offsets, buf, s)
        data[i] = s
    end

    return states
end

# ------------------------------------------------------------
# Hybrid prep: decorrelation + (pluggable) noise-shaped residuals
# ------------------------------------------------------------

"""
    prepare_hybrid_block(data, memories; init_weight=0, shapeL=0.0, shapeR=0.0)

Run the decorrelator and return:
- residuals        :: Vector{Stereo{T}}  # raw decorrelated residuals
- shaped_residuals :: Vector{Stereo{T}}  # noise-shaped residuals (pre-quant)
- final_states     :: NTuple            # updated predictor states
"""
function prepare_hybrid_block(data::AbstractVector{Stereo{T}},
                              memories;
                              init_weight=0,
                              shapeL::Real=0.0,
                              shapeR::Real=0.0) where {T}

    states = make_states(memories; init_weight)
    offsets, TOTAL = compute_offsets(memories)
    buf = MMatrix{2,TOTAL,T}(zeros(T, 2, TOTAL))

    n = length(data)
    residuals        = Vector{Stereo{T}}(undef, n)
    shaped_residuals = Vector{Stereo{T}}(undef, n)

    errL = zero(T)
    errR = zero(T)

    αL = T(shapeL)
    αR = T(shapeR)

    for i in eachindex(data)
        s_in = data[i]

        res, states = process_chain!(memories, states, offsets, buf, s_in)
        residuals[i] = res

        shapedL = res.l + αL * errL
        shapedR = res.r + αR * errR

        shaped = Stereo{T}(shapedL, shapedR)
        shaped_residuals[i] = shaped

        # placeholder: in a real hybrid encoder, err = (res - dequantized_res)
        errL = shapedL - res.l
        errR = shapedR - res.r
    end

    return residuals, shaped_residuals, states
end

# ------------------------------------------------------------
# Simple scalar quantizer for hybrid experiments
# ------------------------------------------------------------

"""
    quantize_hybrid_block(residuals; qL, qR)

Given residuals (Vector{Stereo{T}}), perform simple per-channel scalar quantization:

- q = round(res / q)
- deq = q * q
- corr = res - deq

Returns:
- quantized      :: Vector{Stereo{Int}}
- dequantized    :: Vector{Stereo{T}}
- correction     :: Vector{Stereo{T}}
"""
function quantize_hybrid_block(residuals::AbstractVector{Stereo{T}};
                               qL::Real,
                               qR::Real) where {T}

    n = length(residuals)

    quantized   = Vector{Stereo{Int}}(undef, n)
    dequantized = Vector{Stereo{T}}(undef, n)
    correction  = Vector{Stereo{T}}(undef, n)

    qL_T = T(qL)
    qR_T = T(qR)

    for i in eachindex(residuals)
        r = residuals[i]

        ql = round(Int, r.l / qL_T)
        qr = round(Int, r.r / qR_T)

        dqL = T(ql) * qL_T
        dqR = T(qr) * qR_T

        corrL = r.l - dqL
        corrR = r.r - dqR

        quantized[i]   = Stereo{Int}(ql, qr)
        dequantized[i] = Stereo{T}(dqL, dqR)
        correction[i]  = Stereo{T}(corrL, corrR)
    end

    return quantized, dequantized, correction
end

# ------------------------------------------------------------
# Full hybrid block: decorrelate + shape + quantize
# ------------------------------------------------------------

"""
    hybrid_block(data, memories; init_weight=0, qL, qR, shapeL=0.0, shapeR=0.0)

High-level hybrid prep:

1. Decorrelate input `data` → residuals
2. Noise-shape residuals (pre-quant)
3. Quantize shaped residuals

Returns:
- residuals        :: Vector{Stereo{T}}   # raw decorrelated residuals
- shaped_residuals :: Vector{Stereo{T}}   # shaped (pre-quant) residuals
- quantized        :: Vector{Stereo{Int}} # lossy residuals (to entropy-code later)
- dequantized      :: Vector{Stereo{T}}   # reconstructed residuals
- correction       :: Vector{Stereo{T}}   # correction residuals
- final_states     :: NTuple              # predictor states
"""
function hybrid_block(data::AbstractVector{Stereo{T}},
                      memories;
                      init_weight=0,
                      qL::Real,
                      qR::Real,
                      shapeL::Real=0.0,
                      shapeR::Real=0.0) where {T}

    residuals, shaped_residuals, states =
        prepare_hybrid_block(data, memories;
                             init_weight=init_weight,
                             shapeL=shapeL,
                             shapeR=shapeR)

    quantized, dequantized, correction =
        quantize_hybrid_block(shaped_residuals; qL=qL, qR=qR)

    return residuals, shaped_residuals, quantized, dequantized, correction, states
end

end # module
