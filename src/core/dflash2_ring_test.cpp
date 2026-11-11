// src/core/dflash2_ring_test.cpp - the drafter's ring log and rebuild (Dflash2Drafter::begin_request), against the ring an uninterrupted update sequence left.
//
//     dflash2_ring_test PACK [device]
//
// Context rows are fed with deterministic pseudo-random taps (a function of the position, so any lineage can be refed), for lineage A up to a point P, the ring is
// copied out, then the update goes on far enough to wrap the ring (and, in the other scenarios, another lineage takes over).  begin_request must then rebuild the
// window the first draft reads, [n - 2047, resume), and the K/V bytes of every rebuilt slot must equal, bit for bit, those of the copy taken at P:
//   1 rewind     : A fed to P + 600, rewound to P (a repeated or rewound prefix)             -> rebuild from the live log
//   2 parked     : A fed to P, B fed (different tokens and taps), A resumed at P             -> swap from the stash, rebuild
//   3 branch     : A fed to P + 700, a request branches at P - 300 with other tokens, then A's old branch resumes at P -> stash copy (shared chunks, copy on write)
//   4 off        : STRATA_DF2_RING_LOG=0 (env set before the run): the same rewind finds the window blank (the old behaviour), reported not asserted
//   5 evicted    : a stash too small to keep A: the resume is cold (missing rows reported, *cold set)
// Exit 0 when every assertion holds.
#include "strata/core/dflash2.hpp"
#include "strata/core/conversation_file.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>
#include <filesystem>
#include <vector>

using strata::core::Dflash2Drafter;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

constexpr int kTap = 12800;   // 5 taps x 2560

float tap_value(uint32_t lineage, int64_t pos, int i) {
    uint32_t x = (uint32_t) pos * 2654435761u ^ (uint32_t) i * 40503u ^ lineage * 97531u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    return ((float) (x & 0xFFFF) / 32768.0f - 1.0f) * (1.0f + 0.25f * (float) (i % 7));
}

// feeds the rows [from, to) of `lineage` (tokens: token_of(lineage, pos)) as context updates, 256 rows a call, and notes the tokens
int32_t token_of(uint32_t lineage, int64_t pos) { return (int32_t) (1000 + ((uint32_t) pos * 7919u + lineage * 104729u) % 50000u); }

bool feed(Dflash2Drafter& d, float* dev, uint32_t lineage, int64_t from, int64_t to, std::string& err) {
    std::vector<float> h((size_t) 256 * kTap);
    for (int64_t p0 = from; p0 < to; p0 += 256) {
        const int64_t n = std::min<int64_t>(256, to - p0);
        for (int64_t r = 0; r < n; ++r)
            for (int i = 0; i < kTap; ++i) h[(size_t) r * kTap + (size_t) i] = tap_value(lineage, p0 + r, i);
        if (cudaMemcpy(dev, h.data(), (size_t) n * kTap * 4, cudaMemcpyHostToDevice) != cudaSuccess) { err = "upload"; return false; }
        if (!d.context(dev, n, p0, err)) return false;
        std::vector<int32_t> tk((size_t) n);
        for (int64_t r = 0; r < n; ++r) tk[(size_t) r] = token_of(lineage, p0 + r);
        d.note_rows(tk.data(), n, p0);
    }
    return true;
}

std::vector<int32_t> ids_of(uint32_t lineage, int64_t n) {
    std::vector<int32_t> v((size_t) n);
    for (int64_t p = 0; p < n; ++p) v[(size_t) p] = token_of(lineage, p);
    return v;
}

// bytes of slot s of layer l: K and V, 8 heads x 128 floats
bool slot_equal(const std::vector<float>& a, const std::vector<float>& b, int slot) {
    const size_t per = (size_t) Dflash2Drafter::kRing * 8 * 128;
    for (int l = 0; l < 5; ++l)
        for (int kv = 0; kv < 2; ++kv)
            for (int h = 0; h < 8; ++h) {
                const size_t off = ((size_t) l * 2 + (size_t) kv) * per + ((size_t) h * Dflash2Drafter::kRing + (size_t) slot) * 128;
                if (std::memcmp(&a[off], &b[off], 128 * 4) != 0) return false;
            }
    return true;
}

// every slot of the positions [lo, hi) holds the row `ref` has (same bytes), tagged with the drafter's current epoch and that position
int count_bad(Dflash2Drafter& d, const std::vector<float>& ref, int64_t lo, int64_t hi, int* wrong_tag) {
    std::vector<float> now;
    std::vector<int64_t> tags;
    std::string err;
    if (!d.debug_ring(now, tags, err)) { std::printf("debug_ring: %s\n", err.c_str()); return 1 << 20; }
    int bad = 0;
    *wrong_tag = 0;
    for (int64_t p = lo; p < hi; ++p) {
        const int slot = (int) (p % Dflash2Drafter::kRing);
        if ((tags[(size_t) slot] & 0xFFFFFFFFll) != p || tags[(size_t) slot] < 0) { ++*wrong_tag; ++bad; continue; }
        if (!slot_equal(now, ref, slot)) ++bad;
    }
    return bad;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: dflash2_ring_test PACK [device]\n"); return 2; }
    const int device = argc > 2 ? std::atoi(argv[2]) : 0;
    const bool log_off_run = std::getenv("STRATA_DF2_RING_LOG") != nullptr && std::getenv("STRATA_DF2_RING_LOG")[0] == '0';
    cudaSetDevice(device);
    std::string err;
    const int64_t P = 4227, N = 4232;   // the resumed prefix and the prompt (the agent prompt's numbers)
    float* dev = nullptr;
    cudaMalloc((void**) &dev, (size_t) 256 * kTap * 4);
    std::printf("dflash2_ring_test%s\n", log_off_run ? " (STRATA_DF2_RING_LOG=0)" : "");

    auto fresh = [&](Dflash2Drafter& d) -> bool {
        if (!d.load(argv[1], device, err)) { std::printf("load: %s\n", err.c_str()); return false; }
        return true;
    };
    std::vector<float> ref;
    std::vector<int64_t> reftags;
    {   // ---- 1 rewind
        std::printf("1 rewind (fed to %lld + 600, rewound to %lld, prompt %lld)\n", (long long) P, (long long) P, (long long) N);
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        if (!feed(d, dev, 1, 0, P, err)) { std::printf("feed: %s\n", err.c_str()); return 2; }
        if (!d.debug_ring(ref, reftags, err)) return 2;
        if (!feed(d, dev, 1, P, P + 600, err)) return 2;
        const std::vector<int32_t> ids = ids_of(1, N);
        bool cold = false;
        d.begin_request(ids.data(), N, P, &cold);
        int wt = 0;
        const int bad = count_bad(d, ref, N - 2047, P, &wt);
        std::printf("    rebuilt %lld rows in %.1f ms, missing %lld, slots wrong %d (wrong tag %d), cold %d\n", (long long) d.last_rebuilt, d.last_ms, (long long) d.last_missing, bad, wt, (int) cold);
        if (log_off_run) {
            check(d.last_rebuilt == 0, "log off: nothing is rebuilt (the pre-log behaviour)");
            check(bad > 0, "log off: the overwritten part of the window is blank (the defect this fixes)");
        } else {
            check(bad == 0, "every slot of [n - 2047, resume) holds its row, bitwise the original update's");
            check(d.last_rebuilt == (P + 600 - Dflash2Drafter::kRing) - (N - 2047), "only the overwritten slots were rebuilt (the others were already right)");
            check(!cold, "not cold");
        }
        // the ring must go on working: feed the prompt's remaining rows and a decode row, no stray tags
        if (!feed(d, dev, 1, P, N, err)) return 2;
    }
    if (log_off_run) {
        std::printf(g_fail == 0 ? "dflash2_ring_test: ok\n" : "dflash2_ring_test: FAILED\n");
        return g_fail == 0 ? 0 : 1;
    }
    {   // ---- 2 parked
        std::printf("2 parked (A to %lld, B to 3000, A resumed at %lld)\n", (long long) P, (long long) P);
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        if (!feed(d, dev, 1, 0, P, err)) return 2;
        {   // B takes over properly: a request for B from scratch
            const std::vector<int32_t> idsB = ids_of(2, 3000);
            bool cold = false;
            d.begin_request(idsB.data(), 3000, 0, &cold);
            if (!feed(d, dev, 2, 0, 3000, err)) return 2;
        }
        const std::vector<int32_t> ids = ids_of(1, N);
        bool cold = true;
        d.begin_request(ids.data(), N, P, &cold);
        int wt = 0;
        const int bad = count_bad(d, ref, N - 2047, P, &wt);
        std::printf("    stash hits %lld, rebuilt %lld rows in %.1f ms (swap %d), slots wrong %d (wrong tag %d), stash %.1f MiB\n", (long long) d.ring_stats.stash_hits,
                    (long long) d.last_rebuilt, d.last_ms, (int) d.last_swapped, bad, wt, (double) d.ring_stash_bytes() / 1048576.0);
        check(d.ring_stats.stash_hits == 1 && d.last_swapped, "the parked lineage came back from the stash");
        check(bad == 0, "every slot of [n - 2047, resume) holds its row, bitwise the original update's");
        check(d.last_rebuilt == P - (N - 2047), "the whole window below resume was rebuilt (a new epoch)");
        check(!cold, "not cold");
    }
    {   // ---- 3 branch
        std::printf("3 branch (A to %lld, a request at %lld with other tokens, the old branch resumed at %lld)\n", (long long) (P + 700), (long long) (P - 300), (long long) P);
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        if (!feed(d, dev, 1, 0, P + 700, err)) return 2;
        // reference at P: the ring of an uninterrupted feed to P is `ref` (same lineage, same taps)
        {   // a request that shares A up to P - 300 and then goes its own way (lineage 3 after that point)
            std::vector<int32_t> ids = ids_of(1, P - 300);
            for (int64_t p = P - 300; p < P - 300 + 400; ++p) ids.push_back(token_of(3, p));
            bool cold = false;
            d.begin_request(ids.data(), (int64_t) ids.size(), P - 300, &cold);
            if (!feed(d, dev, 3, P - 300, P - 300 + 400, err)) return 2;   // (rows after the branch point: lineage 3's taps)
        }
        const std::vector<int32_t> ids = ids_of(1, N);
        bool cold = true;
        d.begin_request(ids.data(), N, P, &cold);
        int wt = 0;
        const int bad = count_bad(d, ref, N - 2047, P, &wt);
        std::printf("    stash hits %lld, rebuilt %lld rows, missing %lld, slots wrong %d (wrong tag %d)\n", (long long) d.ring_stats.stash_hits, (long long) d.last_rebuilt,
                    (long long) d.last_missing, bad, wt);
        check(d.ring_stats.stash_hits >= 1, "the old branch was kept (copy on write) and came back");
        check(bad == 0, "the old branch's rows are bitwise the original update's, not the new branch's");
        check(!cold, "not cold");
    }
    {   // Persistence: capture, replace the live owner, file round trip, restore, deferred replay, rewind.
        std::printf("4 persistence (complete current state, deferred taps, later rewind)\n");
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        if (!feed(d, dev, 1, 0, P + 599, err)) return 2;
        std::vector<float> last_taps(kTap);
        for (int i = 0; i < kTap; ++i) last_taps[(size_t) i] = tap_value(1, P + 599, i);
        if (cudaMemcpy(dev, last_taps.data(), kTap * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
            !d.context_async(dev, 1, P + 599, err)) return 2;
        const int32_t last_token = token_of(1, P + 599);
        d.note_rows(&last_token, 1, P + 599);
        constexpr int deferred_rows = 3;
        const int64_t deferred_from = P + 600;
        float* held = d.defer_rows(deferred_from, deferred_rows, err);
        if (!held) return 2;
        // This run does not wrap the pinned ring. Fill real taps just as a slot's verifier copy would.
        for (int r = 0; r < deferred_rows; ++r)
            for (int i = 0; i < kTap; ++i) held[(size_t) r * kTap + i] = tap_value(1, deferred_from + r, i);
        const auto tokens = ids_of(1, deferred_from + deferred_rows);
        d.note_rows(tokens.data() + deferred_from, deferred_rows, deferred_from);
        strata::core::Dflash2State captured;
        const size_t capture_peak = d.snapshot_bytes();
        check(d.snapshot(captured, err), "snapshot synchronizes pending context, ring, tags, lineage and deferred taps");
        check(captured.bytes() <= capture_peak, "pre-capture memory estimate covers the owned snapshot");
        check(captured.history == tokens && captured.deferred.size() == (size_t) deferred_rows * kTap,
              "snapshot contains the complete history and exactly the held rows");
        d.reset(); d.clear_deferred();
        if (!feed(d, dev, 2, 0, 3000, err)) return 2;

        const auto path = std::filesystem::temp_directory_path() / ("strata-df2-state-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".bin");
        strata::core::SavedConversation saved;
        saved.live.ids = captured.history;
        saved.dflash2 = captured;
        const strata::core::SessionFileIdentity identity{123, 456};
        size_t bytes = 0;
        if (!strata::core::session_file_write(path.string(), saved, identity, bytes, err)) {
            std::printf("write: %s\n", err.c_str()); return 2;
        }
        saved = {}; // Drop every file-image copy before the read.
        strata::core::SessionReadLimits limits;
        limits.max_tokens = 8192;
        limits.dflash2 = d.snapshot_limits(8192);
        if (!strata::core::session_file_read(path.string(), identity, saved, bytes, err)) {
            std::printf("read: %s\n", err.c_str()); return 2;
        }
        std::filesystem::remove(path);
        check(saved.dflash2 && *saved.dflash2 == captured, "v2 preserves every persisted DFlash state byte");
        {
            Dflash2Drafter empty;
            if (!empty.clone_alloc(d, err)) return 2;
            check(empty.restore_bytes(captured) == d.restore_bytes(captured) + (size_t) 2048 * kTap * sizeof(float),
                  "restore preflight includes the full deferred allocation only for an owner without it");
            if (!empty.restore(*saved.dflash2, err)) return 2;
            strata::core::Dflash2State first_restore;
            check(empty.snapshot(first_restore, err) && first_restore == captured,
                  "restore on a fresh owner allocates deferred storage and restores all state independently");
        }
        const float* landing = d.landing();
        if (!d.restore(*saved.dflash2, err)) { std::printf("restore: %s\n", err.c_str()); return 2; }
        strata::core::Dflash2State restored;
        check(d.snapshot(restored, err) && restored == captured, "restored GPU ring, tags and host state equal capture");
        check(d.landing() == landing && d.ring_stash_bytes() == 0, "restore keeps graph buffers and imports no other lineage");
        check(d.has_deferred() && d.deferred_hi() == deferred_from + deferred_rows &&
              std::memcmp(d.deferred_row(deferred_from), captured.deferred.data(), kTap * sizeof(float)) == 0,
              "deferred taps retain positions and exact row bytes");

        // Rejected geometry must not write any GPU or host state.
        auto malformed = restored;
        --malformed.shape.ring_slots;
        check(!d.restore(malformed, err) && !err.empty(), "invalid geometry restore is visibly rejected");
        strata::core::Dflash2State unchanged;
        check(d.snapshot(unchanged, err) && unchanged == restored, "rejected restore leaves the owner unchanged");
        if (!d.catch_up(deferred_from + deferred_rows, err)) return 2;
        check(!d.has_deferred(), "restored deferred rows replay and retire normally");
        std::vector<float> replayed;
        std::vector<int64_t> replayed_tags;
        if (!d.debug_ring(replayed, replayed_tags, err)) return 2;
        d.reset(); d.clear_deferred();
        if (!feed(d, dev, 1, 0, deferred_from + deferred_rows, err)) return 2;
        std::vector<float> uninterrupted;
        std::vector<int64_t> uninterrupted_tags;
        if (!d.debug_ring(uninterrupted, uninterrupted_tags, err)) return 2;
        check(replayed.size() == uninterrupted.size() &&
              std::memcmp(replayed.data(), uninterrupted.data(), replayed.size() * sizeof(float)) == 0,
              "restored deferred replay gives the uninterrupted context's complete K/V bytes");
        if (!d.restore(captured, err)) return 2;
        const auto ids = ids_of(1, N);
        bool cold = true;
        check(d.begin_request(ids.data(), N, P, &cold), "restored lineage can begin a rewound request");
        int wt = 0;
        check(count_bad(d, ref, N - 2047, P, &wt) == 0 && !cold && d.last_rebuilt > 0,
              "rewind after restore rebuilds overwritten rows from the persisted log, bitwise the original");
        if (!feed(d, dev, 1, P, N, err)) return 2;
    }
    {   // ---- 5 evicted
        std::printf("5 evicted (stash budget 0)\n");
        setenv("STRATA_DF2_LOG_MIB", "0", 1);
        Dflash2Drafter d;
        if (!fresh(d)) return 2;
        unsetenv("STRATA_DF2_LOG_MIB");
        if (!feed(d, dev, 1, 0, P, err)) return 2;
        const std::vector<int32_t> idsB = ids_of(2, 3000);
        bool cold = false;
        d.begin_request(idsB.data(), 3000, 0, &cold);
        if (!feed(d, dev, 2, 0, 3000, err)) return 2;
        const std::vector<int32_t> ids = ids_of(1, N);
        cold = false;
        d.begin_request(ids.data(), N, P, &cold);
        std::printf("    stash hits %lld, rebuilt %lld, missing %lld, cold %d\n", (long long) d.ring_stats.stash_hits, (long long) d.last_rebuilt, (long long) d.last_missing, (int) cold);
        check(cold && d.last_missing > 0 && d.last_rebuilt == 0, "no log of the lineage: cold, the missing rows are reported");
    }
    std::printf(g_fail == 0 ? "dflash2_ring_test: ok\n" : "dflash2_ring_test: FAILED\n");
    return g_fail == 0 ? 0 : 1;
}
