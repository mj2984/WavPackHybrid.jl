###############################
# StereoDecision.jl
###############################

# Tuple-based decorrelation spec:
# joint_stereo :: Bool
# deltas       :: NTuple{N,Int32}
# terms        :: NTuple{N,Int}
###############################

struct DecorrSpec{N}
    joint_stereo::Bool
    deltas::NTuple{N,Int32}
    terms::NTuple{N,Int}
end

# --------------------------------
# Build specs from WavpackDecorrelationTables
# Each entry: (joint::Int, delta::Int, terms::NTuple{N,Int})
# --------------------------------

function load_specs_from_tables(
    tables::Dict{Symbol,Vector},
    key::Symbol,
)
    raw = tables[key]
    specs = DecorrSpec[]
    for (joint, delta, terms) in raw
        N = length(terms)
        tnt = ntuple(i -> Int(terms[i]), N)
        dnt = ntuple(i -> Int32(delta), N)
        push!(specs, DecorrSpec{N}(joint != 0, dnt, tnt))
    end
    return specs
end

# Example usage (in your main code):
# const DEFAULT_SPECS_JL = load_specs_from_tables(WavpackDecorrelationTables, :default)
# const FAST_SPECS_JL    = load_specs_from_tables(WavpackDecorrelationTables, :fast)

# --------------------------------
# Mid/side transform (JS trial)
# --------------------------------

function to_joint_stereo(data::Vector{Stereo{Int32}})
    out = similar(data)
    @inbounds for i in eachindex(data)
        L = data[i].l
        R = data[i].r
        mid  = Int32(L - R)
        side = Int32(R + (mid >>> 1))
        out[i] = Stereo{Int32}(mid, side)
    end
    return out
end

# --------------------------------
# log2 "size" of residual buffer
# --------------------------------

@inline function log2buffer(res::Vector{Stereo{Int32}}; log_limit::Int32 = 0)
    acc = Int64(0)
    @inbounds for s in res
        for v in (s.l, s.r)
            x = abs(Int64(v)) + 1
            if x == 0
                continue
            end
            bits = 63 - leading_zeros(x)
            frac = bits == 0 ? 0 : ((x << (63 - bits)) >>> 55)
            val = Int32(bits * 256 + (frac & 0xff))
            if log_limit != 0 && val > log_limit
                val = log_limit
            end
            acc += val
        end
    end
    return acc
end

# --------------------------------
# Single decorrelation trial
# --------------------------------

function trial_chain(
    data::Vector{Stereo{Int32}},
    memories;
    init_weight::Int32 = 0,
)
    states = make_states(memories; init_weight)
    bufs   = make_buffers(memories)

    n = length(data)
    res = Vector{Stereo{Int32}}(undef, n)

    s = Stereo{Int32}(0, 0)
    @inbounds for i in 1:n
        s = data[i]
        s, states = process_chain!(memories, states, bufs, s)
        res[i] = s
    end

    return res
end

# --------------------------------
# Stereo mode chooser (TS vs JS)
# --------------------------------
# Returns:
#   samples  :: modified in-place to best residuals
#   memories :: best decorrelation chain (NTuple)
#   use_js   :: Bool, whether JOINT_STEREO should be set
# --------------------------------

function choose_stereo_mode!(
    samples::Vector{Stereo{Int32}},
    specs::Vector{DecorrSpec};
    num_passes::Int = 1,
    init_weight::Int32 = 0,
    force_js::Bool = false,
    force_ts::Bool = false,
)

    n = length(samples)

    all_zero = true
    @inbounds for s in samples
        if s.l != 0 || s.r != 0
            all_zero = false
            break
        end
    end
    if all_zero
        return samples, nothing, false
    end

    best_size = typemax(Int64)
    best_res  = Vector{Stereo{Int32}}(undef, n)
    best_mem  = nothing
    best_js   = false

    js_buf = nothing

    for pass in 1:num_passes
        for spec in specs
            use_js = force_js || (spec.joint_stereo && !force_ts)

            trial_in = if use_js
                js_buf === nothing && (js_buf = to_joint_stereo(samples))
                js_buf
            else
                samples
            end

            memories = make_memories(Int32, spec.terms, spec.deltas)
            res = trial_chain(trial_in, memories; init_weight)

            size = log2buffer(res)

            if size < best_size
                best_size = size
                best_res  = res
                best_mem  = memories
                best_js   = use_js
            end
        end
    end

    @inbounds for i in 1:n
        samples[i] = best_res[i]
    end

    return samples, best_mem, best_js
end
