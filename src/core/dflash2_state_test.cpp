// Standalone CUDA/HIP state regression: no downloaded model or drafter pack.
// Builds a one-layer context-only DF2P with identity/select Q8 projections and runs the real SDK and kernels.
// This exercises persistence, owner transfer, deferred replay and rewind, not model inference or block proposals.
#include "strata/core/conversation_file.hpp"
#include "strata/core/dflash2.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
constexpr int kHidden = 2560;
constexpr int kHeads = 8;
constexpr int kHeadDim = 128;

void check(bool ok, const char* label, const std::string& error = {}) {
    if (!ok) throw std::runtime_error(std::string(label) + (error.empty() ? "" : ": " + error));
    std::printf("ok: %s\n", label);
}

struct TempDir {
    fs::path path = fs::temp_directory_path() / ("strata-df2-state-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDir() { fs::create_directories(path); }
    ~TempDir() { std::error_code error; fs::remove_all(path, error); }
};

struct DeviceTaps {
    float* p = nullptr;
    DeviceTaps() { check(cudaMalloc((void**) &p, 8 * kHidden * sizeof(float)) == cudaSuccess, "allocate tap input"); }
    ~DeviceTaps() { if (p) cudaFree(p); }
};

struct FixtureTensor {
    std::string name, dtype;
    int rows = 0, cols = 0;
    std::vector<uint8_t> bytes;
};

FixtureTensor select_q8(const std::string& name, int rows, int cols, int shift = 0) {
    FixtureTensor t{name, "q8_0", rows, cols, std::vector<uint8_t>((size_t) rows * cols / 32 * 34)};
    // Literal GGUF Q8_0 blocks: fp16 scale 1, all codes zero except one unit coefficient per matrix row.
    for (size_t i = 0; i < t.bytes.size(); i += 34) { t.bytes[i] = 0x00; t.bytes[i + 1] = 0x3c; }
    for (int r = 0; r < rows; ++r) {
        const int col = (r + shift) % cols;
        t.bytes[((size_t) r * (cols / 32) + col / 32) * 34 + 2 + col % 32] = 1;
    }
    return t;
}

FixtureTensor norm(const std::string& name, int n) {
    FixtureTensor t{name, "f32", n, 1, std::vector<uint8_t>((size_t) n * 4)};
    for (size_t i = 0; i < t.bytes.size(); i += 4) {
        t.bytes[i] = 0x00; t.bytes[i + 1] = 0x00; t.bytes[i + 2] = 0x80; t.bytes[i + 3] = 0x3f;
    }
    return t;
}

void make_context_pack(const fs::path& path) {
    std::vector<FixtureTensor> tensors;
    tensors.push_back(select_q8("fc.weight", 2560, 2560));
    tensors.push_back(norm("hidden_norm.weight", 2560));
    tensors.push_back(select_q8("layers.0.self_attn.k_proj.weight", 1024, 2560));
    tensors.push_back(select_q8("layers.0.self_attn.v_proj.weight", 1024, 2560, 1024));
    tensors.push_back(norm("layers.0.self_attn.k_norm.weight", 128));
    // eps=0 makes uniform +/- powers-of-two normalize exactly to +/-1; the window has the runtime's 2048 rows.
    std::string index = "cfg hidden 2560\ncfg layers 1\ncfg heads 32\ncfg kv_heads 8\ncfg head_dim 128\n"
                        "cfg inter 7680\ncfg window 2048\ncfg conv_group 16\ncfg rank 256\ncfg mask_token 0\n"
                        "cfg vocab 50000\ncfg rope_theta 10000\ncfg eps 0\ncfg taps 0\n";
    std::vector<uint8_t> payload;
    for (const auto& t : tensors) {
        payload.resize((payload.size() + 255) / 256 * 256, 0);
        index += "t " + t.name + " " + t.dtype + " " + std::to_string(t.rows) + " " + std::to_string(t.cols) +
                 " " + std::to_string(payload.size()) + " " + std::to_string(t.bytes.size()) + "\n";
        payload.insert(payload.end(), t.bytes.begin(), t.bytes.end());
    }
    std::ofstream f(path, std::ios::binary);
    const uint32_t version = 1, index_bytes = (uint32_t) index.size();
    f.write("DF2P", 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&index_bytes), 4);
    f.write(index.data(), (std::streamsize) index.size());
    f.write(reinterpret_cast<const char*>(payload.data()), (std::streamsize) payload.size());
    check(bool(f), "write context-only synthetic pack");
}

void context_rows(Dflash2Drafter& d, DeviceTaps& dev, int64_t pos, const std::vector<float>& values, bool async = false) {
    std::vector<float> taps(values.size() * kHidden);
    for (size_t r = 0; r < values.size(); ++r)
        std::fill(taps.begin() + r * kHidden, taps.begin() + (r + 1) * kHidden, values[r]);
    check(cudaMemcpy(dev.p, taps.data(), taps.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess, "upload real taps");
    std::string error;
    check(async ? d.context_async(dev.p, (int64_t) values.size(), pos, error) :
                  d.context(dev.p, (int64_t) values.size(), pos, error), "run real context projection", error);
}

Dflash2State capture(Dflash2Drafter& d) {
    Dflash2State s;
    std::string error;
    check(d.snapshot(s, error), "capture complete real GPU state", error);
    return s;
}

uint32_t bits(float f) { uint32_t b = 0; std::memcpy(&b, &f, 4); return b; }

void value_is(const Dflash2State& s, int64_t position, uint32_t expected) {
    const size_t per = (size_t) Dflash2Drafter::kRing * kHeads * kHeadDim;
    bool equal = true;
    for (int h = 0; h < kHeads; ++h)
        for (int dim = 0; dim < kHeadDim; ++dim) {
            const size_t at = per + ((size_t) h * Dflash2Drafter::kRing + (size_t) (position % Dflash2Drafter::kRing)) * kHeadDim + dim;
            equal = equal && bits(s.ring[at]) == expected;
        }
    check(equal, "all V bytes equal independent literal +/-1");
}

bool equal_slot(const Dflash2State& actual, const Dflash2State& expected, int slot) {
    if (actual.tags[(size_t) slot] != expected.tags[(size_t) slot]) {
        std::fprintf(stderr, "tag mismatch slot=%d actual=0x%016llx expected=0x%016llx epochs=%lld/%lld\n", slot,
                     (unsigned long long) actual.tags[(size_t) slot], (unsigned long long) expected.tags[(size_t) slot],
                     (long long) actual.epoch, (long long) expected.epoch);
        return false;
    }
    const size_t per = (size_t) Dflash2Drafter::kRing * kHeads * kHeadDim;
    for (int kv = 0; kv < 2; ++kv)
        for (int h = 0; h < kHeads; ++h)
            for (int dim = 0; dim < kHeadDim; ++dim) {
                const size_t at = (size_t) kv * per + ((size_t) h * Dflash2Drafter::kRing + slot) * kHeadDim + dim;
                if (bits(actual.ring[at]) == bits(expected.ring[at])) continue;
                std::fprintf(stderr, "ring mismatch slot=%d part=%c head=%d dim=%d actual=0x%08x (%.9g) expected=0x%08x (%.9g)\n",
                             slot, kv == 0 ? 'K' : 'V', h, dim, bits(actual.ring[at]), actual.ring[at],
                             bits(expected.ring[at]), expected.ring[at]);
                return false;
            }
    return true;
}

bool export_is_private(const Dflash2State& s) {
    const size_t per = (size_t) Dflash2Drafter::kRing * kHeads * kHeadDim;
    if ((uint64_t) s.deferred_hi > s.history.size()) {
        std::fprintf(stderr, "export retains deferred suffix lo=%lld hi=%lld history=%zu\n",
                     (long long) s.deferred_lo, (long long) s.deferred_hi, s.history.size());
        return false;
    }
    for (size_t slot = 0; slot < s.tags.size(); ++slot) {
        const int64_t tag = s.tags[slot];
        if (tag != -1) {
            const uint64_t pos = (uint64_t) tag & 0xffffffffull;
            if ((tag >> 32) == s.epoch && pos < s.history.size() &&
                !(pos >= (uint64_t) s.deferred_lo && pos < (uint64_t) s.deferred_hi)) continue;
            std::fprintf(stderr, "export retains non-current tag slot=%zu tag=0x%016llx epoch=%lld history=%zu\n",
                         slot, (unsigned long long) tag, (long long) s.epoch, s.history.size());
            return false;
        }
        if (s.slot_positions[slot] != -1) return false;
        for (int64_t layer = 0; layer < s.shape.layers; ++layer)
            for (int kv = 0; kv < 2; ++kv)
                for (int head = 0; head < kHeads; ++head)
                    for (int dim = 0; dim < kHeadDim; ++dim) {
                        const size_t at = ((size_t) layer * 2 + kv) * per + ((size_t) head * Dflash2Drafter::kRing + slot) * kHeadDim + dim;
                        if (bits(s.ring[at]) != 0) {
                            std::fprintf(stderr, "export retains invalid ring bytes slot=%zu part=%c head=%d dim=%d bits=0x%08x\n",
                                         slot, kv == 0 ? 'K' : 'V', head, dim, bits(s.ring[at]));
                            return false;
                        }
                    }
    }
    for (const auto& c : s.log)
        for (size_t row = 0; row < c.valid.size(); ++row) {
            const uint64_t pos = c.index * 256 + row;
            if (c.valid[row]) {
                if (pos >= s.history.size() ||
                    (pos >= (uint64_t) s.deferred_lo && pos < (uint64_t) s.deferred_hi)) return false;
                continue;
            }
            for (size_t i = 0; i < 2560; ++i)
                if (c.q[row * 2560 + i] != 0) {
                    std::fprintf(stderr, "export retains invalid log code position=%llu feature=%zu code=%d\n",
                                 (unsigned long long) pos, i, (int) c.q[row * 2560 + i]);
                    return false;
                }
            for (size_t i = 0; i < 80; ++i) if (bits(c.scales[row * 80 + i]) != 0) return false;
        }
    return true;
}

Dflash2State private_capture(Dflash2Drafter& owner) {
    std::string error;
    std::vector<float> before, after;
    std::vector<int64_t> tags_before, tags_after;
    check(owner.debug_ring(before, tags_before, error), "read actual ring before export", error);
    Dflash2State s = capture(owner);
    check(owner.debug_ring(after, tags_after, error), "read actual ring after export", error);
    check(tags_before == tags_after && before.size() == after.size() &&
          std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) == 0,
          "snapshot normalization never mutates the live device ring or tags");
    check(export_is_private(s), "export contains only current-lineage ring and valid log-row bytes");
    return s;
}

Dflash2State persist(Dflash2Drafter& owner, const Dflash2State& state, const fs::path& path) {
    SavedConversation image;
    image.live.ids = state.history;
    image.dflash2 = state;
    std::string error;
    size_t bytes = 0;
    const SessionFileIdentity identity{0x1234, 0x5678};
    check(session_file_write(path.string(), image, identity, bytes, error), "persist private current-lineage snapshot", error);
    image = {};
    SessionReadLimits limits;
    limits.max_tokens = 4096;
    limits.dflash2 = owner.snapshot_limits(4096);
    check(session_file_read(path.string(), identity, image, bytes, error, limits), "read private current-lineage snapshot", error);
    check(image.dflash2 && *image.dflash2 == state && export_is_private(*image.dflash2),
          "persisted bytes preserve current lineage without stale ring or invalid log contents");
    return std::move(*image.dflash2);
}

void set_ring_log(bool enabled) {
#ifdef _WIN32
    _putenv_s("STRATA_DF2_RING_LOG", enabled ? "1" : "0");
#else
    setenv("STRATA_DF2_RING_LOG", enabled ? "1" : "0", 1);
#endif
}

void uniform_context(Dflash2Drafter& owner, DeviceTaps& dev, int from, int to, float value) {
    std::vector<float> rows(8 * kHidden, value);
    check(cudaMemcpy(dev.p, rows.data(), rows.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess,
          "upload uniform context rows");
    std::string error;
    bool ok = true;
    for (int pos = from; pos < to && ok; pos += 8)
        ok = owner.context(dev.p, std::min(8, to - pos), pos, error);
    check(ok, "project the actual context run", error);
}

void request_transition(Dflash2Drafter& weights, DeviceTaps& taps, const fs::path& dir, bool log_on, int resume) {
    std::printf("request transition: log=%d resume=%d\n", (int) log_on, resume);
    set_ring_log(log_on);
    Dflash2Drafter owner, reference_owner;
    std::string error;
    check(owner.clone_alloc(weights, error) && reference_owner.clone_alloc(weights, error),
          "allocate transition and independent reference owners", error);
    check(owner.ring_log_on() == log_on && reference_owner.ring_log_on() == log_on, "exercise actual requested log mode");
    constexpr int n = 2066, tail = n - 2047;
    std::vector<int32_t> a(n), b(n);
    for (int i = 0; i < n; ++i) { a[(size_t) i] = 100 + i; b[(size_t) i] = i < resume ? a[(size_t) i] : 5000 + i; }
    owner.note_rows(a.data(), n, 0);
    uniform_context(owner, taps, 0, n, 1.0f); // A physically fills all 2064 ring slots.
    if (resume) {
        float* old_suffix = owner.defer_rows(n - 1, 1, error);
        check(old_suffix != nullptr, "stage A's pending suffix before branching", error);
        std::fill(old_suffix, old_suffix + kHidden, 64.0f);
    }
    bool cold = true;
    check(owner.begin_request(b.data(), n, resume, &cold) && !cold, "switch requests through normal begin_request, without reset");
    owner.note_rows(b.data() + resume, n - resume, resume);
    uniform_context(owner, taps, tail, n, -2.0f); // Production prefill reads only the final 2047 prompt taps.
    const Dflash2State actual = private_capture(owner);
    size_t valid = 0;
    bool ownership = true, values = true;
    const size_t per = (size_t) Dflash2Drafter::kRing * kHeads * kHeadDim;
    for (int pos = n - Dflash2Drafter::kRing; pos < n; ++pos) {
        const size_t slot = (size_t) (pos % Dflash2Drafter::kRing);
        const bool current = pos < resume || pos >= tail;
        const int64_t wanted = current ? (actual.epoch << 32) | pos : -1;
        valid += actual.tags[slot] != -1 ? 1 : 0;
        if (actual.tags[slot] != wanted || actual.slot_positions[slot] != wanted) {
            if (ownership)
                std::fprintf(stderr, "ownership mismatch log=%d resume=%d pos=%d actual=0x%016llx shadow=0x%016llx expected=0x%016llx\n",
                             (int) log_on, resume, pos, (unsigned long long) actual.tags[slot],
                             (unsigned long long) actual.slot_positions[slot], (unsigned long long) wanted);
            ownership = false;
        }
        if (!current) continue;
        const uint32_t wanted_v = pos < resume ? 0x3f800000u : 0xbf800000u;
        for (int h = 0; h < kHeads; ++h)
            for (int dim = 0; dim < kHeadDim; ++dim)
                values = values && bits(actual.ring[per + ((size_t) h * Dflash2Drafter::kRing + slot) * kHeadDim + dim]) == wanted_v;
    }
    const size_t wanted_rows = 2047 + (resume > 2 ? (size_t) resume - 2 : 0);
    if (valid != wanted_rows)
        std::fprintf(stderr, "ownership row count log=%d resume=%d actual=%zu expected=%zu\n", (int) log_on, resume, valid, wanted_rows);
    check(ownership && valid == wanted_rows, "export retains exactly B's rows and the physically present shared prefix");
    check(values, "every retained V is the independent literal B=-1 or common-prefix A=+1");
    check(!owner.has_deferred() && actual.deferred.empty(), "request ownership boundary discards A's pending suffix");
    reference_owner.note_rows(b.data(), n, 0);
    if (resume) uniform_context(reference_owner, taps, 0, resume, 1.0f);
    uniform_context(reference_owner, taps, tail, n, -2.0f);
    const Dflash2State reference = capture(reference_owner);
    for (size_t slot = 0; slot < actual.tags.size(); ++slot)
        if (!equal_slot(actual, reference, (int) slot)) check(false, "transition K/V differs from the independent fresh B context");
    check(actual == reference, "all B K/V, ownership tags, token history and log bytes equal independent fresh context");
    const fs::path path = dir / ("transition-" + std::to_string(log_on) + "-" + std::to_string(resume) + ".bin");
    const Dflash2State from_file = persist(owner, actual, path);
    check(reference_owner.restore(from_file, error) && capture(reference_owner) == actual,
          "request-transition file restores only B and its valid shared prefix", error);
}

void run(int device) {
    set_ring_log(true);
    check(cudaSetDevice(device) == cudaSuccess, "select GPU");
    TempDir temp;
    const fs::path pack = temp.path / "context.pack", file = temp.path / "session.bin";
    make_context_pack(pack);
    Dflash2Drafter live;
    std::string error;
    check(live.load(pack.string(), device, error), "load real synthetic context weights", error);
    check(live.ring_log_on(), "real ring log enabled");
    Dflash2Drafter slot;
    check(slot.clone_alloc(live, error), "allocate independent physical slot", error);
    DeviceTaps taps;
    std::vector<int32_t> ids(2066);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) (100 + i);
    // A sparse context fixture: token identity covers the conversation; only rows used by this rewind are projected.
    live.note_rows(ids.data(), (int64_t) ids.size(), 0);
    context_rows(live, taps, 0, {1, -2, 4, -8, 1, -2, 4, -8}, true);
    const Dflash2State reference = capture(live); // snapshot must finish the pending async projection and log flush
    check(reference.tags[0] == 0 && reference.tags[1] == 1 && reference.history == ids, "literal tags and full token history");
    check(reference.log.size() == 1 && reference.log[0].q[0] == 127 && reference.log[0].q[2560] == -127 &&
          bits(reference.log[0].scales[0]) == 0x3c010204u, "independent literal Q8 log codes and 1/127 scale bytes");
    value_is(reference, 0, 0x3f800000u);
    value_is(reference, 1, 0xbf800000u);
    context_rows(live, taps, 2065, {4}); // real write wraps slot 1, so the later rewind must rebuild from the saved log
    context_rows(live, taps, 99, {-4}); // A row that conversation B will not overwrite, exposing stale-epoch leakage.
    const float deferred_values[] = {-2, 4, -8};
    for (int r = 0; r < 3; ++r) {
        float* dst = live.defer_rows(2047 + r, 1, error);
        check(dst != nullptr, "defer a real row across the pinned-ring wrap", error);
        std::fill(dst, dst + kHidden, deferred_values[r]);
    }
    const size_t need = live.snapshot_bytes();
    const Dflash2State original = capture(live);
    check(original.bytes() <= need && original.deferred_lo == 2047 && original.deferred_hi == 2050 &&
          original.deferred[0] == -2 && original.deferred[2560] == 4 && original.deferred[5120] == -8,
          "capture owns bounded chronological deferred bytes, including both sides of the wrap");
    value_is(original, 2065, 0x3f800000u);
    live.reset();
    check(!live.has_deferred() && live.deferred_hi() == 0, "reset invalidates old-lineage pending taps without caller cleanup");
    const std::vector<int32_t> ids_b{900, 901, 902, 903, 904, 905, 906, 907};
    live.note_rows(ids_b.data(), (int64_t) ids_b.size(), 0);
    context_rows(live, taps, 0, {-1, 2, -4, 8, -1, 2, -4, 8});
    const Dflash2State b = private_capture(live);
    check(b.tags[99] == -1 && b.history == ids_b, "A-to-B export drops A's untouched slot and preserves B's token identity");
    const Dflash2State b_disk = persist(live, b, temp.path / "b.bin");
    check(slot.restore(b_disk, error) && capture(slot) == b, "A-to-B file restores B without importing A's leftover bytes", error);
    check(!(b == original), "source owner actually overwritten before serialization");

    SavedConversation image;
    image.live.ids = original.history;
    image.dflash2 = original;
    const SessionFileIdentity identity{0x1234, 0x5678};
    size_t bytes = 0;
    check(session_file_write(file.string(), image, identity, bytes, error), "save actual GPU snapshot as manual v2", error);
    image = {};
    SessionReadLimits limits;
    limits.max_tokens = 4096;
    limits.dflash2 = slot.snapshot_limits(4096);
    check(session_file_read(file.string(), identity, image, bytes, error, limits), "read bounded v2 state", error);
    check(image.dflash2 && *image.dflash2 == original, "every serialized state field retains its bytes");
    const float* landing = slot.landing();
    check(slot.restore(*image.dflash2, error), "restore into the physical slot", error);
    check(slot.landing() == landing && capture(slot) == original, "restore keeps captured buffer addresses and exact state");
    check(live.copy_state_from(slot, error), "return restored slot state to main owner through D2D fast path", error);
    check(capture(live) == original, "slot-to-main transfer includes all wrapped deferred taps and counters");
    {
        Dflash2Drafter fresh;
        check(fresh.clone_alloc(live, error), "allocate fresh destination owner", error);
        check(fresh.copy_state_from(slot, error), "owner copy allocates missing deferred storage", error);
        check(capture(fresh) == original, "fresh owner copy retains complete state");
        Dflash2Drafter empty;
        check(empty.clone_alloc(live, error), "allocate empty source owner", error);
        check(fresh.copy_state_from(empty, error) && !fresh.has_deferred(), "empty source clears destination deferred range", error);
    }
    float* changed = slot.defer_rows(2047, 1, error);
    check(changed != nullptr, "overwrite source deferred storage after owner transfer", error);
    std::fill(changed, changed + kHidden, 32.0f);
    check(capture(live) == original, "copied deferred taps do not borrow source storage");

    auto invalid = original;
    --invalid.shape.ring_slots;
    check(!live.restore(invalid, error) && !error.empty(), "reject invalid geometry before writes");
    check(capture(live) == original, "rejected restore leaves real state unchanged");
    check(live.catch_up(2050, error) && !live.has_deferred(), "replay restored deferred rows with real context kernels", error);
    const Dflash2State replayed = capture(live);
    value_is(replayed, 2047, 0xbf800000u);
    value_is(replayed, 2048, 0x3f800000u);
    value_is(replayed, 2049, 0xbf800000u);
    bool cold = true;
    check(live.begin_request(ids.data(), 2048, 7, &cold) && !cold && live.last_rebuilt == 1 && live.last_missing == 0,
          "rewind really rebuilds the overwritten row from the persisted lineage log");
    for (int row = 0; row < 3; ++row) {
        float* suffix = live.defer_rows(8 + row, 1, error);
        check(suffix != nullptr, "stage an unpublished suffix after rewind for bounded CPU export", error);
        std::fill(suffix, suffix + kHidden, 16.0f * (float) (1 << row));
    }
    const Dflash2State rewound = private_capture(live);
    check(rewound.history == std::vector<int32_t>(ids.begin(), ids.begin() + 7) &&
          rewound.log.size() == 1 && rewound.log[0].valid[7] == 0,
          "rewind truncates token identity and log validity, retaining only the current lineage");
    check(rewound.deferred.empty() && rewound.deferred_lo == 0 && rewound.deferred_hi == 0 &&
          live.has_deferred() && live.deferred_hi() == 11 && live.deferred_row(8)[0] == 16.0f,
          "rewound export excludes the old pending suffix without changing live deferred taps");
    for (int pos = 1; pos < 7; ++pos)
        check(equal_slot(rewound, reference, pos) && rewound.tags[(size_t) pos] == pos, "rewound K/V and tags equal uninterrupted real context");
    value_is(rewound, 1, 0xbf800000u);
    const Dflash2State rewound_disk = persist(live, rewound, temp.path / "rewound.bin");
    Dflash2Drafter parked;
    check(parked.clone_alloc(live, error) && parked.restore(rewound_disk, error),
          "restore the private rewound file into another owner", error);
    context_rows(live, taps, 7, {-4});
    live.note_rows(ids.data() + 7, 1, 7);
    const Dflash2State continued = capture(live);
    check(continued.history.size() == 8 && continued.history[7] == 107 && continued.tags[7] == 7, "continued request appends correct history and ring tag");
    value_is(continued, 7, 0xbf800000u);
    context_rows(parked, taps, 7, {-4});
    parked.note_rows(ids.data() + 7, 1, 7);
    check(capture(parked) == continued, "persisted rewind continues with byte-identical state");
    const Dflash2State source = capture(slot);
    check(source.history == original.history && source.log == original.log, "owner logs remain copy-on-write during replay and rewind");
    for (bool log_on : {true, false})
        for (int resume : {0, 8}) request_transition(live, taps, temp.path, log_on, resume);
}
} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::fprintf(stderr, "SKIP: no CUDA/HIP device\n");
        return 77;
    }
    try {
        run(argc > 1 ? std::atoi(argv[1]) : 0);
        std::printf("dflash2_state_test: ok (context/state only; no model inference)\n");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dflash2_state_test: %s\n", error.what());
        return 1;
    }
}
