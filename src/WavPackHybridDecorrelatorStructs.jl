module WavPackHybridDecorrelators

using StaticArrays
using SamplesCore: Stereo

# ------------------------------------------------------------
# Pass memory types
# ------------------------------------------------------------

struct IntraPassMemoryGeneric{N,T}
    delta::Int
    offset::Int
end

struct IntraPassMemorySpecial{N,T}
    delta::Int
    offset::Int
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
# Intra-channel processing (using single MMatrix buffer)
# ------------------------------------------------------------

@inline function process_pass!(
    mem::IntraPassMemoryGeneric{N,T},
    st::IntraPassState,
    buf::MMatrix{2,TOTAL,T},
    s::Stereo{T},
) where {N,T,TOTAL}

    L = s.l
    R = s.r

    bi = mem.offset + st.idx

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
    s::Stereo{T},
) where {N,T,TOTAL}

    L = s.l
    R = s.r

    bi = mem.offset + st.idx

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

# unify arity for generated chain
@inline process_pass!(
    mem::IntraPassMemoryCrossChannel{term},
    st::CrossPassState,
    buf::MMatrix{2,TOTAL,T},
    s::Stereo{T},
) where {T,term,TOTAL} = process_pass!(mem, st, s)

# ------------------------------------------------------------
# Constructors for memories and states
# ------------------------------------------------------------

function make_memories(::Type{T},
                       terms::AbstractVector{Int},
                       deltas::AbstractVector{Int}) where {T}

    mems = ()
    offset = 0

    for (term, delta) in zip(terms, deltas)
        if 1 <= term <= 8
            N = term
            mem = IntraPassMemoryGeneric{N,T}(delta, offset)
            offset += N

        elseif term == 17
            N = 1
            mem = IntraPassMemorySpecial{N,T}(delta, offset)
            offset += N

        elseif term == 18
            N = 2
            mem = IntraPassMemorySpecial{N,T}(delta, offset)
            offset += N

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
# Generated, fully unrolled chain
# ------------------------------------------------------------

@generated function process_chain!(
    memories::MT,
    states::ST,
    buf::MMatrix{2,TOTAL,T},
    s::Stereo{T},
) where {N,MT<:NTuple{N,Any},ST<:NTuple{N,Any},TOTAL,T}

    steps = Vector{Expr}(undef, N)
    for i in 1:N
        steps[i] = :(s, st_$i = process_pass!(memories[$i], states[$i], buf, s))
    end

    new_states = Expr(:tuple, [Symbol("st_$i") for i in 1:N]...)

    quote
        $(Expr(:block, steps...))
        return s, $new_states
    end
end

# ------------------------------------------------------------
# Buffer size helper
# ------------------------------------------------------------

function buffer_total(memories)
    total = 0
    for mem in memories
        if mem isa IntraPassMemoryGeneric
            total = max(total, mem.offset + mem.delta)
        elseif mem isa IntraPassMemorySpecial
            total = max(total, mem.offset + mem.delta)
        end
    end
    return max(total, 1)
end

# ------------------------------------------------------------
# Block processor on Vector{Stereo{T}}
# ------------------------------------------------------------

function process_block!(data::AbstractVector{Stereo{T}},
                        memories;
                        init_weight=0) where {T}

    states = make_states(memories; init_weight)
    TOTAL = buffer_total(memories)

    buf = MMatrix{2,TOTAL,T}(zeros(T, 2, TOTAL))

    for i in eachindex(data)
        s = data[i]
        s, states = process_chain!(memories, states, buf, s)
        data[i] = s
    end

    return states
end

end
