// CPU-owned state of one DFlash2 conversation. No weights, graphs, device addresses or stashed lineages.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core {

struct Dflash2Shape {
    int64_t hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0;
    int64_t window = 0, group = 0, rank = 0, mask_token = 0, vocab = 0;
    int64_t ring_slots = 0, deferred_slots = 0, log_chunk_rows = 0;
    float theta = 0, eps = 0;
    bool q4 = false;
    std::vector<int32_t> taps;
    bool operator==(const Dflash2Shape&) const = default;
};

struct Dflash2LogChunk {
    uint64_t index = 0;
    std::vector<int8_t> q;
    std::vector<float> scales;
    std::vector<uint8_t> valid;
    bool operator==(const Dflash2LogChunk&) const = default;
};

struct Dflash2SnapshotLimits {
    Dflash2Shape shape;
    uint64_t max_tokens = 0;
    bool log_enabled = false;
};

struct Dflash2State {
    Dflash2Shape shape;
    bool log_enabled = false;
    int64_t epoch = 0, have_ctx = 0, ring_hi = 0;
    // Layer-major, per layer K then V, each [kv_heads][ring_slots][head_dim].
    // Snapshot exports zero rows outside the current lineage and uses -1 for their tags; live buffers are untouched.
    std::vector<float> ring;
    std::vector<int64_t> tags, slot_positions;
    std::vector<int32_t> history;
    // Only the current lineage's allocated chunks, in strictly increasing index order.
    // Snapshot exports zero codes/scales for invalid rows, including a truncated or deferred suffix.
    std::vector<Dflash2LogChunk> log;
    int64_t deferred_lo = 0, deferred_hi = 0;
    // Chronological rows [lo, hi), each [taps.size()][hidden]; not the unused pinned ring.
    std::vector<float> deferred;

    size_t bytes() const {
        size_t n = shape.taps.capacity() * sizeof(int32_t) + ring.capacity() * sizeof(float) +
                   (tags.capacity() + slot_positions.capacity()) * sizeof(int64_t) +
                   history.capacity() * sizeof(int32_t) + log.capacity() * sizeof(Dflash2LogChunk) +
                   deferred.capacity() * sizeof(float);
        for (const auto& c : log) n += c.q.capacity() + c.scales.capacity() * sizeof(float) + c.valid.capacity();
        return n;
    }
    bool operator==(const Dflash2State&) const = default;
};

// Fixed bounds of this implementation, independent of a file's declared dimensions. The runtime still checks
// the complete shape against its loaded pack before restore. These checks precede all variable-size parsing.
inline bool dflash2_shape_validate(const Dflash2Shape& s, std::string& error) {
    if (s.hidden != 2560 || s.layers < 0 || s.layers > 8 || s.heads != 32 || s.kv_heads != 8 || s.head_dim != 128 ||
        s.inter < INT32_MIN || s.inter > INT32_MAX || s.window < INT32_MIN || s.window > 2048 ||
        s.group < INT32_MIN || s.group > INT32_MAX || s.rank != 256 ||
        s.mask_token < INT32_MIN || s.mask_token > INT32_MAX ||
        s.ring_slots != 2064 || s.deferred_slots != 2048 || s.log_chunk_rows != 256) {
        error = "dflash2: invalid snapshot shape";
        return false;
    }
    return true;
}

inline bool dflash2_deferred_elements(const Dflash2Shape& s, uint64_t rows, uint64_t& elements) {
    const uint64_t max = std::numeric_limits<size_t>::max() / sizeof(float);
    if (s.hidden <= 0 || rows > max / (uint64_t) s.hidden) return false;
    elements = rows * (uint64_t) s.hidden;
    if (elements && s.taps.size() > max / elements) return false;
    elements *= s.taps.size();
    return true;
}

inline bool dflash2_state_validate(const Dflash2State& s, std::string& error,
                                  uint64_t max_tokens = INT32_MAX) {
    if (!dflash2_shape_validate(s.shape, error)) return false;
    max_tokens = std::min<uint64_t>(max_tokens, INT32_MAX);
    auto fail = [&](const char* why) { error = std::string("dflash2: ") + why; return false; };
    const auto& g = s.shape;
    const size_t ring = (size_t) g.layers * 2 * (size_t) g.kv_heads * (size_t) g.ring_slots * (size_t) g.head_dim;
    if (s.epoch < 0 || s.epoch > INT32_MAX || s.have_ctx < 0 || (uint64_t) s.have_ctx > max_tokens ||
        s.ring_hi < 0 || (uint64_t) s.ring_hi > max_tokens || s.history.size() > max_tokens ||
        s.ring.size() != ring || s.tags.size() != (size_t) g.ring_slots || s.slot_positions.size() != s.tags.size())
        return fail("invalid snapshot counters or ring size");
    auto valid_tag = [&](int64_t t, size_t slot) {
        if (t == -1) return true;
        if (t < 0 || (t >> 32) > s.epoch) return false;
        const uint64_t p = (uint64_t) t & 0xffffffffull;
        return p < max_tokens && p % (uint64_t) g.ring_slots == slot;
    };
    for (size_t i = 0; i < s.tags.size(); ++i)
        if (!valid_tag(s.tags[i], i) || !valid_tag(s.slot_positions[i], i)) return fail("invalid snapshot ring tag");
    const uint64_t chunks = (max_tokens + (uint64_t) g.log_chunk_rows - 1) / (uint64_t) g.log_chunk_rows;
    if ((!s.log_enabled && !s.log.empty()) || s.log.size() > chunks) return fail("invalid snapshot log count");
    uint64_t previous = 0;
    for (size_t i = 0; i < s.log.size(); ++i) {
        const auto& c = s.log[i];
        if (c.index >= chunks || (i && c.index <= previous) ||
            c.q.size() != (size_t) g.log_chunk_rows * (size_t) g.hidden ||
            c.scales.size() != (size_t) g.log_chunk_rows * (size_t) (g.hidden / 32) ||
            c.valid.size() != (size_t) g.log_chunk_rows) return fail("invalid snapshot log chunk");
        for (size_t r = 0; r < c.valid.size(); ++r)
            if (c.valid[r] > 1 || (c.valid[r] && c.index * (uint64_t) g.log_chunk_rows + r >= max_tokens))
                return fail("invalid snapshot logged position");
        previous = c.index;
    }
    uint64_t deferred_elements = 0;
    if (s.deferred_lo < 0 || s.deferred_hi < s.deferred_lo || (uint64_t) s.deferred_hi > max_tokens ||
        s.deferred_hi - s.deferred_lo > g.deferred_slots ||
        (s.deferred_hi == s.deferred_lo && s.deferred_lo != 0) ||
        !dflash2_deferred_elements(g, (uint64_t) (s.deferred_hi - s.deferred_lo), deferred_elements) ||
        s.deferred.size() != deferred_elements)
        return fail("invalid snapshot deferred taps");
    return true;
}

} // namespace strata::core
