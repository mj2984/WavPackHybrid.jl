###############################
# StereoDecision.jl
###############################

# Requires:
# - Stereo{T}
# - apply_weight
# - nosend_word
# - make_memories, make_states, make_buffers, process_chain!
# - nbits_table, log2_table from WavPackHybridTables.jl

############################################################
# 1. Decorrelation Spec
############################################################

struct DecorrSpec{N}
    joint_stereo::Bool
    delta::Int32
    terms::NTuple{N,Int}
end

function load_specs_from_tables(tables::Dict{Symbol,Vector}, key::Symbol)
    raw = tables[key]
    specs = DecorrSpec[]
    for (joint, delta, terms) in raw
        N = length(terms)
        tnt = ntuple(i -> Int(terms[i]), N)
        push!(specs, DecorrSpec{N}(joint != 0, Int32(delta), tnt))
    end
    return specs
end

############################################################
# 2. wp_log2 (bit-exact)
############################################################

@inline function wp_log2(avalue::UInt32)
    av = avalue + (avalue >> 9)

    if av < 0x100
        dbits = nbits_table[Int(av) + 1]
        idx   = Int((av << (9 - dbits)) & 0xff) + 1
        return (UInt32(dbits) << 8) + UInt32(log2_table[idx])
    else
        dbits::Int
        if av < 0x10000
            dbits = nbits_table[Int(av >> 8) + 1] + 8
        elseif av < 0x1000000
            dbits = nbits_table[Int(av >> 16) + 1] + 16
        else
            dbits = nbits_table[Int(av >> 24) + 1] + 24
        end

        idx = Int((av >> (dbits - 9)) & 0xff) + 1
        return (UInt32(dbits) << 8) + UInt32(log2_table[idx])
    end
end

############################################################
# 3. log2buffer (bit-exact)
############################################################

@inline function log2buffer(res::Vector{Stereo{Int32}}; limit::UInt32 = 0)
    acc = UInt32(0)

    @inbounds for s in res
        for v in (s.l, s.r)
            av = UInt32(abs(v))
            val = wp_log2(av)

            if limit != 0 && val >= limit
                return typemax(UInt32)   # matches C: return (uint32_t)-1
            end

            acc += val
        end
    end

    return acc
end

############################################################
# 4. Mid/Side Transform
############################################################

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

############################################################
# 5. Noisy Buffer (bit-exact stereo_add_noise)
############################################################

function stereo_add_noise!(wps, noisy::Vector{Stereo{Int32}},
                           orig::Vector{Stereo{Int32}})
    flags = wps.wphdr.flags
    hybrid_shape = (flags & HYBRID_SHAPE) != 0
    is_new       = (flags & NEW_SHAPING) != 0

    shaping_array = wps.dc.shaping_array

    error0 = Int32(0)
    error1 = Int32(0)

    cnt = length(orig)

    if hybrid_shape
        acc0 = wps.dc.shaping_acc[1]
        acc1 = wps.dc.shaping_acc[2]
        δ0   = wps.dc.shaping_delta[1]
        δ1   = wps.dc.shaping_delta[2]

        idx_sa = 1

        @inbounds for i in 1:cnt
            s = orig[i]

            # ----- channel 0 -----
            shaping_weight0 = shaping_array !== nothing ?
                Int32(shaping_array[idx_sa]) :
                begin acc0 += δ0; acc0 >> 16 end

            temp0 = -apply_weight(shaping_weight0, error0)
            q0    = nosend_word(wps, s.l, 0)
            base_err0 = q0 - s.l

            if is_new && shaping_weight0 < 0 && temp0 != 0
                if temp0 == error0
                    temp0 += temp0 < 0 ? 1 : -1
                end
                error0 = base_err0 + temp0
            else
                error0 = base_err0 + temp0
            end

            noisyL = s.l + error0

            # ----- channel 1 -----
            shaping_weight1 = begin acc1 += δ1; acc1 >> 16 end

            temp1 = -apply_weight(shaping_weight1, error1)
            q1    = nosend_word(wps, s.r, 1)
            base_err1 = q1 - s.r

            if is_new && shaping_weight1 < 0 && temp1 != 0
                if temp1 == error1
                    temp1 += temp1 < 0 ? 1 : -1
                end
                error1 = base_err1 + temp1
            else
                error1 = base_err1 + temp1
            end

            noisyR = s.r + error1

            noisy[i] = Stereo{Int32}(noisyL, noisyR)

            if shaping_array !== nothing
                idx_sa += 1
            end
        end

        if shaping_array === nothing
            acc0 -= δ0 * cnt
            acc1 -= δ1 * cnt
        end

        wps.dc.shaping_acc = (acc0, acc1)

    else
        @inbounds for i in 1:cnt
            s = orig[i]
            q0 = nosend_word(wps, s.l, 0)
            q1 = nosend_word(wps, s.r, 1)
            noisy[i] = Stereo{Int32}(s.l + (q0 - s.l),
                                     s.r + (q1 - s.r))
        end
    end

    return noisy
end

############################################################
# 6. Single decorrelation trial
############################################################

function trial_chain(data::Vector{Stereo{Int32}}, memories; init_weight::Int32 = 0)
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

############################################################
# 7. Stereo Mode Chooser (now uses noisy buffer)
############################################################

function choose_stereo_mode!(
    wps,
    orig_samples::Vector{Stereo{Int32}},
    specs::Vector{DecorrSpec};
    num_passes::Int = 1,
    init_weight::Int32 = 0,
    force_js::Bool = false,
    force_ts::Bool = false,
)

    n = length(orig_samples)

    # Build noisy buffer (bit-exact)
    noisy = similar(orig_samples)
    stereo_add_noise!(wps, noisy, orig_samples)

    # If all zero, skip
    all_zero = all(s -> s.l == 0 && s.r == 0, noisy)
    if all_zero
        return noisy, nothing, false
    end

    best_size = typemax(UInt64)
    best_res  = Vector{Stereo{Int32}}(undef, n)
    best_mem  = nothing
    best_js   = false

    js_buf = nothing

    for pass in 1:num_passes
        for spec in specs
            use_js = force_js || (spec.joint_stereo && !force_ts)

            trial_in = if use_js
                js_buf === nothing && (js_buf = to_joint_stereo(noisy))
                js_buf
            else
                noisy
            end

            N = length(spec.terms)
            deltas = ntuple(_ -> spec.delta, N)
            memories = make_memories(Int32, spec.terms, deltas)
            res = trial_chain(trial_in, memories; init_weight)

            size = UInt64(log2buffer(res))

            if size < best_size
                best_size = size
                best_res  = res
                best_mem  = memories
                best_js   = use_js
            end
        end
    end

    return best_res, best_mem, best_js
end
