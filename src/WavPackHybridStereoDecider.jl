###############################
# StereoDecision.jl
###############################
# Assumes:
# - WavpackStream has fields:
#     w::WordsData
#     dc.shaping_acc::NTuple{2,Int32}
#     dc.shaping_delta::NTuple{2,Int32}
#     dc.shaping_array::Union{Nothing,AbstractVector}
#     wphdr.flags::UInt32

struct MedianParams
    div0::UInt32
    div1::UInt32
    div2::UInt32
end
const MEDIANS = MedianParams(128, 64, 32)
struct SlowLevelParams
    shift::UInt32
    offset::UInt32
end
const SLOW = SlowLevelParams(8, 1 << 7)

mutable struct EntropyData
    median::NTuple{3,UInt32}   # (m0, m1, m2)
    error_limit::UInt32
    slow_level::UInt32
end

mutable struct WordsData
    bitrate_delta::NTuple{2,UInt32}
    bitrate_acc::NTuple{2,UInt32}
    pend_data::UInt32
    holding_one::UInt32
    zeros_acc::UInt32
    holding_zero::Int32
    pend_count::Int32
    c::NTuple{2,EntropyData}
end

@inline get_med(m::UInt32) = (m >> 4) + 1
@inline inc_med(m::UInt32, div::UInt32) = m + ((m + div) ÷ div) * 5
@inline dec_med(m::UInt32, div::UInt32) = m - ((m + (div - 2)) ÷ div) * 2

# 4. update_error_limit! (stub for now)
function update_error_limit!(wps)
    # TODO: port from C:
    #   void update_error_limit (WavpackStream *wps)
    return
end

function nosend_word(wps, value::Int32, chan::Int)
    wd = wps.w::WordsData
    c  = wd.c[chan + 1]

    # sign folding
    sign = value < 0
    v = sign ? ~value : value

    # hybrid-lossless only: always update error limit on chan 0
    if chan == 0
        update_error_limit!(wps)
    end

    m0, m1, m2 = c.median

    # --- median partitioning ---
    if v < Int32(get_med(m0))
        low  = UInt32(0)
        high = get_med(m0) - 1
        m0   = dec_med(m0, MEDIANS.div0)
    else
        low  = get_med(m0)
        m0   = inc_med(m0, MEDIANS.div0)

        dv = v - Int32(low)

        if dv < Int32(get_med(m1))
            high = low + get_med(m1) - 1
            m1   = dec_med(m1, MEDIANS.div1)
        else
            low += get_med(m1)
            m1   = inc_med(m1, MEDIANS.div1)

            dv2 = v - Int32(low)

            if dv2 < Int32(get_med(m2))
                high = low + get_med(m2) - 1
                m2   = dec_med(m2, MEDIANS.div2)
            else
                ones = UInt32(2 + dv2 ÷ Int32(get_med(m2)))
                low  += (ones - 2) * get_med(m2)
                high  = low + get_med(m2) - 1
                m2    = inc_med(m2, MEDIANS.div2)
            end
        end
    end

    c.median = (m0, m1, m2)

    # --- error-limit refinement ---
    mid = (high + low + 1) >> 1

    if c.error_limit != 0
        while high - low > c.error_limit
            if v < Int32(mid)
                high = mid - 1
            else
                low = mid
            end
            mid = (high + low + 1) >> 1
        end
    else
        mid = UInt32(v)
    end

    # --- slow-level update ---
    c.slow_level -= (c.slow_level + SLOW.offset) >> SLOW.shift
    c.slow_level += wp_log2(mid)

    return sign ? ~Int32(mid) : Int32(mid)
end

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

function stereo_add_noise!(wps, noisy::Vector{Stereo{Int32}},
                           orig::Vector{Stereo{Int32}})
    flags = wps.wphdr.flags
    is_new = (flags & NEW_SHAPING) != 0

    shaping_array = wps.dc.shaping_array

    error0 = Int32(0)
    error1 = Int32(0)

    cnt = length(orig)

    # hybrid lossless only: always use shaping path
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

    return noisy
end

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

    noisy = similar(orig_samples)
    stereo_add_noise!(wps, noisy, orig_samples)

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
