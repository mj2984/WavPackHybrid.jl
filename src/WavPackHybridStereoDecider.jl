###############################
# StereoDecision.jl
###############################
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

############################################################
# 2. Entropy / words state
############################################################

mutable struct EntropyData
    median::NTuple{3,UInt32}   # (m0, m1, m2)
    error_limit::UInt32
    slow_level::UInt32
end

mutable struct WordsData
    c::NTuple{2,EntropyData}   # one per channel
end

############################################################
# 3. Median helpers (Julian, pure, readable)
############################################################

@inline get_med(m::UInt32) = (m >> 4) + 1

@inline inc_med(m::UInt32, div::UInt32) =
    m + ((m + div) ÷ div) * 5

@inline dec_med(m::UInt32, div::UInt32) =
    m - ((m + (div - 2)) ÷ div) * 2

############################################################
# 4. update_error_limit! (stub for now)
############################################################

function update_error_limit!(wps)
    # TODO: port from C
    return
end

############################################################
# 5. nosend_word (clean, Julian, bit-exact)
############################################################

function nosend_word(wps, value::Int32, chan::Int)
    c = wps.w.c[chan + 1]

    # --- sign folding ---
    sign = value < 0
    v = sign ? ~value : value

    # hybrid-lossless always updates error limit on channel 0
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
                low += (ones - 2) * get_med(m2)
                high = low + get_med(m2) - 1
                m2   = inc_med(m2, MEDIANS.div2)
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

############################################################
# 6. Noisy buffer (clean, Julian)
############################################################

function stereo_add_noise!(wps, noisy, orig)
    acc0, acc1 = wps.dc.shaping_acc
    δ0,   δ1   = wps.dc.shaping_delta
    arr        = wps.dc.shaping_array
    is_new     = (wps.wphdr.flags & NEW_SHAPING) != 0

    e0 = Int32(0)
    e1 = Int32(0)

    @inbounds for i in eachindex(orig)
        s = orig[i]

        # --- channel 0 ---
        w0 = arr === nothing ? (acc0 += δ0; acc0 >> 16) : Int32(arr[i])
        t0 = -apply_weight(w0, e0)
        q0 = nosend_word(wps, s.l, 0)
        base0 = q0 - s.l

        if is_new && w0 < 0 && t0 != 0 && t0 == e0
            t0 += t0 < 0 ? 1 : -1
        end

        e0 = base0 + t0
        L  = s.l + e0

        # --- channel 1 ---
        w1 = (acc1 += δ1) >> 16
        t1 = -apply_weight(w1, e1)
        q1 = nosend_word(wps, s.r, 1)
        base1 = q1 - s.r

        if is_new && w1 < 0 && t1 != 0 && t1 == e1
            t1 += t1 < 0 ? 1 : -1
        end

        e1 = base1 + t1
        R  = s.r + e1

        noisy[i] = Stereo{Int32}(L, R)
    end

    wps.dc.shaping_acc = (acc0, acc1)
    return noisy
end

############################################################
# 7. Decorrelator trial
############################################################

function trial_chain(data, memories)
    states = make_states(memories)
    bufs   = make_buffers(memories)

    out = similar(data)
    @inbounds for i in eachindex(data)
        s = data[i]
        s, states = process_chain!(memories, states, bufs, s)
        out[i] = s
    end
    return out
end

############################################################
# 8. Stereo mode chooser (clean, modern)
############################################################

function choose_stereo_mode!(wps, orig, specs)
    noisy = similar(orig)
    stereo_add_noise!(wps, noisy, orig)

    best_cost = typemax(UInt64)
    best_res  = nothing
    best_mem  = nothing
    best_js   = false

    js_buf = nothing

    for spec in specs
        use_js = spec.joint_stereo

        trial_in = use_js ?
            (js_buf === nothing ? (js_buf = to_joint_stereo(noisy)) : js_buf) :
            noisy

        deltas = ntuple(_ -> spec.delta, length(spec.terms))
        memories = make_memories(Int32, spec.terms, deltas)

        res = trial_chain(trial_in, memories)
        cost = UInt64(log2buffer(res))

        if cost < best_cost
            best_cost = cost
            best_res  = res
            best_mem  = memories
            best_js   = use_js
        end
    end

    return best_res, best_mem, best_js
end
