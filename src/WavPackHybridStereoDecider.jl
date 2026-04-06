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

struct WavPackDecorrelator{N,Stages<:NTuple{N,WavPackDecorrelatorStage},DeltaT<:Union{Int32,NTuple{N,Int32}}}
    midside::Bool
    delta::DeltaT
    stages::Stages
end
function load_decorrelators(tables, key)
    raw = tables[key]
    decs = Vector{WavPackDecorrelator}(undef, length(raw))
    for i in eachindex(raw)
        midside, delta, terms = raw[i]
        stages = make_stages(terms)
        decs[i] = WavPackDecorrelator(midside != 0, delta, stages)
    end
    return decs
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

@inline function stereo_to_midside(input::Stereo{Int32})
    mid::Int32 = input.l - input.r
    side::Int32 = input.r + (mid >>> 1)
    return Stereo{Int32}(mid,side)
end
stereo_to_midside(data::Vector{Stereo{Int32}}) = @inbounds map(stereo_to_midside,data)

@inline function hybrid_shape_sample_nosend!(
    wps,
    sample::Int32,
    error::Int32,
    acc::Int32,
    δ::Int32,
    chan::Int
)
    acc += δ
    shaping_weight = acc >> 16
    
    temp = -apply_weight(shaping_weight, error)
    
    # nosend_word provides the quantized/processed reference
    q = nosend_word(wps, sample, chan)
    base_err = q - sample

    if shaping_weight < 0 && temp != 0
        if temp == error
            temp += temp < 0 ? Int32(1) : Int32(-1)
        end
    end
    
    new_error = base_err + temp
    new_sample = sample + new_error

    return new_sample, new_error, acc
end

function stereo_add_noise!(wps, noisy::Vector{Stereo{Int32}},
                           orig::Vector{Stereo{Int32}})
    # This refactor assumes the case where shaping_array === nothing
    acc0, acc1 = wps.dc.shaping_acc
    δ0, δ1     = wps.dc.shaping_delta
    error0, error1 = Int32(0), Int32(0)

    @inbounds for i in 1:length(orig)
        s = orig[i]

        # Process Left Channel (0)
        noisyL, error0, acc0 = hybrid_shape_sample_nosend!(
            wps, s.l, error0, acc0, δ0, 0
        )

        # Process Right Channel (1)
        noisyR, error1, acc1 = hybrid_shape_sample_nosend!(
            wps, s.r, error1, acc1, δ1, 1
        )

        noisy[i] = Stereo{Int32}(noisyL, noisyR)
    end

    # Subtract the total change (delta * number of samples) to match WavPack's 
    # block-level state management and prevent drift.
    cnt = length(orig)
    wps.dc.shaping_acc = (acc0 - δ0 * cnt, acc1 - δ1 * cnt)

    return noisy
end

function trial_chain!(res_buffer::Vector{Stereo{Int32}},data::Vector{Stereo{Int32}},decorrelator::WavPackDecorrelator;init_weight=0)
    stages = decorrelator.stages
    states = make_states(stages; init_weight)
    bufs   = make_buffers(stages, Int32)
    @inbounds for i in eachindex(data)
        s, states = process_chain!(decorrelator, states, bufs, data[i])
        res_buffer[i] = s
    end
    return res_buffer
end

has_nonzero_samples(orig_samples) = any(s -> s.l != 0 || s.r != 0, orig_samples)
# Note:: force_joint_stereo, force_true_stereo options are removed from runtime. The equivalent can be achieved by directly altering specs before input.
# Note:: The zero checking logic is moved outside. Please run has_nonzero_samples on original samples before running this if you are unsure if the samples are valid.
#        return noisy, nothing, false
function choose_stereo_mode!(wps,orig_samples::Vector{Stereo{T}},decorrelators::Vector{WavPackDecorrelator};num_passes=1,init_weight=0) where {T}
    n = length(orig_samples)
    noisy = similar(orig_samples)
    stereo_add_noise!(wps, noisy, orig_samples)
    js_buf = stereo_to_midside(noisy)
    num_threads = 1  # later: Threads.nthreads()
    bufs = [Vector{Stereo{Int32}}(undef, n) for _ in 1:(num_threads + 1)]
    best_idx = 1
    best_size = typemax(UInt64)
    best_decor = nothing
    for pass in 1:num_passes
        for decor in decorrelators
            # pick ANY buffer that is not the best buffer
            buf_idx = (best_idx == 1) ? 2 : 1 # (for now, num_threads=1, so this picks the other one)
            buf = bufs[buf_idx]
            trial_in = decor.midside ? js_buf : noisy
            trial_chain!(buf, trial_in, decor; init_weight)
            size = UInt64(log2buffer(buf))
            if size < best_size
                best_size = size
                best_idx = buf_idx
                best_decor = decor
            end
        end
    end
return bufs[best_idx], best_decor
