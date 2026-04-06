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
abstract type WavPackDecorrelatorStage end
struct WavPackGenericStage{Delay} <: WavPackDecorrelatorStage end
struct WavPackSpecialStage{Term} <: WavPackDecorrelatorStage end
struct WavPackCrossChannelStage{Term} <: WavPackDecorrelatorStage end

struct IntraPassState
    weight::Int32
    idx::Int
end
struct CrossPassState
    weight::Int32
end

@inline function process_pass!(::WavPackGenericStage{Delay},delta::Int32,st::IntraPassState,buf::MMatrix{2,Delay,T},s::Stereo{T}) where {Delay,T}
    L = s.l; R = s.r

    bi = st.idx
    dL = buf[1, bi]
    dR = buf[2, bi]

    predL = (st.weight * dL) >>> 10
    predR = (st.weight * dR) >>> 10

    resL = L - predL
    resR = R - predR

    new_weight = update_weight(st.weight, delta, dL, resL)
    new_idx    = (st.idx == Delay ? 1 : st.idx + 1)

    buf[1, bi] = L
    buf[2, bi] = R

    return Stereo{T}(resL, resR), IntraPassState(new_weight, new_idx)
end

@inline function process_pass!(::WavPackSpecialStage{Term},delta::Int32,st::IntraPassState,buf::MMatrix{2,2,T},s::Stereo{T}) where {Term,T}
    L = s.l; R = s.r

    i0 = st.idx
    i1 = (i0 == 1 ? 2 : 1)

    xL0 = buf[1, i0]; xL1 = buf[1, i1]
    xR0 = buf[2, i0]; xR1 = buf[2, i1]

    if Term == 17
        samL = 2*xL0 - xL1
        samR = 2*xR0 - xR1
    elseif Term == 18
        samL = xL0 + ((xL0 - xL1) >>> 1)
        samR = xR0 + ((xR0 - xR1) >>> 1)
    else
        error("Invalid special term")
    end

    predL = (st.weight * samL) >>> 10
    predR = (st.weight * samR) >>> 10

    resL = L - predL
    resR = R - predR

    new_weight = update_weight(st.weight, delta, samL, resL)

    buf[1, i1] = xL0
    buf[2, i1] = xR0
    buf[1, i0] = L
    buf[2, i0] = R

    new_idx = (i0 == 2 ? 1 : 2)

    return Stereo{T}(resL, resR), IntraPassState(new_weight, new_idx)
end

@inline function process_pass!(::WavPackCrossChannelStage{Term},delta::Int32,st::CrossPassState,::Nothing,s::Stereo{T}) where {Term,T}
    L = s.l; R = s.r
    if Term == -1
        pred = (st.weight * L) >>> 10
        resR = R - pred
        new_weight = update_weight(st.weight, delta, L, resR)
        return Stereo{T}(L, resR), CrossPassState(new_weight)
    elseif Term == -2
        pred = (st.weight * R) >>> 10
        resL = L - pred
        new_weight = update_weight(st.weight, delta, R, resL)
        return Stereo{T}(resL, R), CrossPassState(new_weight)
    elseif Term == -3
        predL = (st.weight * R) >>> 10
        predR = (st.weight * L) >>> 10
        resL = L - predL
        resR = R - predR
        err = resL + resR
        new_weight = update_weight(st.weight, delta, err, err)
        return Stereo{T}(resL, resR), CrossPassState(new_weight)
    end
end

@generated function make_stages(terms::NTuple{N,Int}) where {N}
    exprs = Vector{Any}(undef, N)
    for i in 1:N
        term = terms[i]
        exprs[i] = if 1 ≤ term ≤ 8
            :(WavPackGenericStage{$term}())
        elseif term == 17
            :(WavPackSpecialStage{17}())
        elseif term == 18
            :(WavPackSpecialStage{18}())
        else
            :(WavPackCrossChannelStage{$term}())
        end
    end
    return :(($(exprs...),))
end

function make_states(stages; init_weight=0)
    states = ()
    for stage in stages
        if stage isa WavPackGenericStage || stage isa WavPackSpecialStage
            states = (states..., IntraPassState(init_weight, 1))
        else
            states = (states..., CrossPassState(init_weight))
        end
    end
    return states
end

function make_buffers(stages, ::Type{T}) where T
    bufs = ()
    for stage in stages
        if stage isa WavPackGenericStage{N} where N
            bufs = (bufs..., MMatrix{2,N,T}(zeros(T, 2, N)))
        elseif stage isa WavPackSpecialStage
            bufs = (bufs..., MMatrix{2,2,T}(zeros(T, 2, 2)))
        else
            bufs = (bufs..., nothing)
        end
    end
    return bufs
end

@generated function process_chain!(decor::WavPackDecorrelator{N,Stages,DeltaT},states::ST, bufs::BT, s::Stereo{T}) where {N,Stages,DeltaT,ST<:NTuple{N,Any},BT<:NTuple{N,Any},T}
    steps = Expr[]
    if DeltaT === Int32
        for i in 1:N
            push!(steps, :(s, st_$i = process_pass!(decor.stages[$i], decor.delta,
                                                   states[$i], bufs[$i], s)))
        end
    else
        for i in 1:N
            push!(steps, :(s, st_$i = process_pass!(decor.stages[$i], decor.delta[$i],
                                                   states[$i], bufs[$i], s)))
        end
    end
    new_states = Expr(:tuple, [Symbol("st_$i") for i in 1:N]...)
    quote
        $(Expr(:block, steps...))
        return s, $new_states
    end
end

@inline function hybrid_shape_sample_lossy!(
    sample::Int32,
    err::Int32,
    shaping_acc::Int32,
    shaping_delta::Int32,
)
    shaping_acc = Int32(shaping_acc + shaping_delta)
    shaping_weight = Int32(shaping_acc >> 16)

    temp = -apply_weight(shaping_weight, err)

    if shaping_weight < 0 && temp != 0
        if temp == err
            temp += temp < 0 ? Int32(1) : Int32(-1)
        end
        err = -sample
        sample = Int32(sample + temp)
    else
        sample = Int32(sample + temp)
        err = -sample
    end

    return sample, err, shaping_acc, Int16(shaping_weight)
end

@inline function quantize_residual(res::Int32, q::Int32)
    half_q = q >>> 1
    qv = (res + half_q) ÷ q
    dq = Int32(qv * q)
    corr = Int32(res - dq)
    return qv, corr
end

function hybrid_block(data::Vector{Stereo{T}}, decorrelator::WavPackDecorrelator;init_weight=0, qL::T, qR::T, shaping_delta_L::T, shaping_delta_R::T) where {T}
    stages, delta = decorrelator.stages, decorrelator.delta
    states = make_states(stages; init_weight)
    bufs   = make_buffers(stages, Int32)

    n = length(data)
    quantized       = Vector{Stereo{Int32}}(undef, n)
    correction      = Vector{Stereo{Int32}}(undef, n)
    shaping_array_L = Vector{Int16}(undef, n)
    shaping_array_R = Vector{Int16}(undef, n)

    errL, errR = Int32(0), Int32(0)
    shaping_acc_L, shaping_acc_R = Int32(0), Int32(0)

    @inbounds for i in 1:n
        L = data[i].l
        R = data[i].r
        L, errL, shaping_acc_L, wL = hybrid_shape_sample_lossy!(L, errL, shaping_acc_L, shaping_delta_L)
        R, errR, shaping_acc_R, wR = hybrid_shape_sample_lossy!(R, errR, shaping_acc_R, shaping_delta_R)
        shaping_array_L[i] = wL
        shaping_array_R[i] = wR
        res, states = process_chain!(decorrelator, states, bufs, Stereo{Int32}(L, R))
        qvL, corrL = quantize_residual(res.l, qL)
        qvR, corrR = quantize_residual(res.r, qR)
        quantized[i]  = Stereo{Int32}(qvL, qvR)
        correction[i] = Stereo{Int32}(corrL, corrR)
        errL = Int32(errL + qvL * qL)
        errR = Int32(errR + qvR * qR)
    end

    return quantized, correction, states, shaping_array_L, shaping_array_R
end
