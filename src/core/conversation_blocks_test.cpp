// CPU persisted-boundary regression tests; no model, GPU, external cache or service.
#include "strata/core/conversation_blocks.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>

using namespace strata::core;
namespace fs = std::filesystem;
namespace {
int checks = 0;
void check(bool ok, const char* name) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", name); std::exit(1); }
}
ConversationBuffer pattern(size_t cells, size_t stride, uint8_t seed) {
    ConversationBuffer out;
    out.resize(cells * stride);
    out.visit(0, out.size(), [&](uint8_t* p, size_t n, size_t at) {
        for (size_t i = 0; i < n; ++i) p[i] = uint8_t(seed + (at + i) / (4096 * stride) * 31 + (at + i) % 251);
        return true;
    });
    return out;
}
SavedConversation sample() {
    SavedConversation s;
    s.geometry[0] = 42; s.layer_hi = 1; s.cvec = false;
    for (int i = 0; i < 8209; ++i) s.live.ids.push_back(i);
    s.live.imgs = {{17, 123}};
    s.live.gdn = {1, 2, 3}; s.live.ple = {4, 5}; s.live.tails = {6};
    s.live.dead = {7, 8}; s.live.block_pos = {9}; s.live.used = 11;
    auto cp = s.live; cp.ids.resize(4096); cp.used = 12;
    s.checkpoints = {cp};
    ConversationKv main;
    main.cells = 12288; main.format = 2; main.heads = 1; main.head_dim = 2; main.page_size = 64;
    main.pooled_rows = 2; main.idx_dim = 1;
    main.k = pattern(12288, 2, 1); main.v = pattern(12288, 3, 70);
    main.k_scale = pattern(12288, 1, 130); main.v_scale = pattern(12288, 1, 195);
    main.pooled = {10, 11, 12, 13};
    s.kv.push_back(main);
    s.kv.push_back(main); // An explicit private draft, byte-identical to the main layer.
    return s;
}
bool same_cp(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}
bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_cp(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size() ||
        a.dflash2 != b.dflash2) return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i) if (!same_cp(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    return b.stage_images.empty();
}
std::vector<char> slurp(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
void spit(const fs::path& path, const std::vector<char>& data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(data.data(), (std::streamsize) data.size());
}
uint64_t file_bytes(const fs::path& directory) {
    uint64_t total = 0;
    for (const auto& file : fs::recursive_directory_iterator(directory)) if (file.is_regular_file()) total += file.file_size();
    return total;
}
fs::path only_manifest(const fs::path& directory) {
    for (const auto& file : fs::directory_iterator(directory / "manifests")) return file.path();
    return {};
}
void rehash_envelope(std::vector<char>& data) {
    const uint64_t hash = session_hash64(data.data() + 64, data.size() - 80, 0);
    std::memcpy(data.data() + data.size() - 16, &hash, 8);
}
}
int main() {
    const char* tmp = std::getenv("TMPDIR");
    const fs::path root = (tmp ? fs::path(tmp) : fs::temp_directory_path()) /
        ("conversation-blocks-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const SessionFileIdentity id{0x123, 0x456};
    SessionBlockStore store((root / "cache").string(), id);
    const fs::path ns = fs::path(store.namespace_directory());
    const fs::path legacy = root / "cache" / "strata-session-blocks-v1-0000000000000123-0000000000000456";
    fs::create_directories(legacy / "manifests");
    spit(legacy / "manifests" / "legacy.session", {'k', 'e', 'e', 'p'});
    check(ns.filename().string().find("strata-session-blocks-v2-") == 0, "new format has independent namespace");
    std::string error;
    SessionStatus status;
    SessionBlockSummary stats;
    std::vector<SessionBlockEntry> entries;
    check(store.entries(entries, stats, error) && entries.empty() && stats.physical_bytes == 0, "empty discovery");
    const auto a = sample();
    auto b = a;
    b.live.ids.back() = -7; b.live.gdn = {42, 43}; b.live.imgs.push_back({8200, 456});
    b.checkpoints[0].ple = {81};
    b.kv[0].k.visit(8192 * 2, 1, [](uint8_t* p, size_t, size_t) { *p = 77; return true; });
    b.kv[0].pooled = {98, 99};
    b.kv[1].v.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p = 33; return true; });
    size_t bytes_a = 0, bytes_b = 0;
    SessionWriteOptions staged;
    staged.durable = false;
    check(store.write("a", a, {1}, bytes_a, error, staged, &status) && status.published, "first nondurable publish");
    check(store.entries(entries, stats, error) && stats.entries == 1 && stats.blocks == 8 && stats.reused_bytes == 0,
          "eight actual main-part chunks, private draft");
    const uint64_t expected_shared = 2 * 4096 * 7 + 8 * 80;
    check(entries[0].shared_bytes == expected_shared && entries[0].kv_layers == 2 &&
          entries[0].layout.main_kv_layers == 1 && entries[0].live.ids == a.live.ids &&
          entries[0].live.imgs == a.live.imgs && entries[0].checkpoints[0].ids.size() == 4096 && !entries[0].cvec,
          "lightweight metadata and explicit capture shape");
    SessionWriteOptions durable;
    size_t file_flushes = 0;
    durable.phase = [&](const char* phase, uint64_t) { if (std::strcmp(phase, "flush") == 0) ++file_flushes; };
    check(store.write("b", b, {1}, bytes_b, error, durable), "durable branch publish");
    check(file_flushes == stats.blocks + 1, "durable owner flushes every reused chunk before its manifest");
    check(store.entries(entries, stats, error) && stats.blocks == 8 && stats.entries == 2 &&
          stats.reused_bytes == expected_shared && stats.physical_bytes == file_bytes(ns) &&
          stats.logical_bytes == bytes_a + bytes_b && stats.logical_bytes - stats.physical_bytes == expected_shared,
          "identical payloads physically stored once across manifests");
    const uint64_t physical_budget = file_bytes(ns);
    check(store.prune(physical_budget, 2, stats, error) && stats.entries == 2 &&
          stats.logical_bytes > physical_budget && stats.physical_bytes == physical_budget,
          "physical quota retains both owners when their logical bytes exceed unique storage");
    SessionBlockStore reopened((root / "cache").string(), id);
    SavedConversation restored;
    size_t read_bytes = 0;
    check(reopened.read("a", restored, read_bytes, error) && same(a, restored) && read_bytes == bytes_a, "reopen original exact bytes");
    check(reopened.read("b", restored, read_bytes, error) && same(b, restored), "private tails recurrent indexer draft and images exact");
    check(store.write("b", b, {1}, bytes_b, error) && store.entries(entries, stats, error) && stats.entries == 2 &&
          stats.blocks == 8 && stats.physical_bytes == file_bytes(ns), "replacement idempotent");
    check(store.erase("a", error) && reopened.read("b", restored, read_bytes, error) && same(b, restored) &&
          store.entries(entries, stats, error) && stats.entries == 1 && stats.blocks == 8, "shared blocks survive first owner removal");
    check(store.erase("a", error), "erase missing idempotent");
    check(store.erase("b", error) && store.entries(entries, stats, error) && stats.entries == 0 && stats.blocks == 0 &&
          file_bytes(ns) == 0, "last owner releases all managed chunks");

    check(store.write("a", a, {1}, bytes_a, error), "failure fixture");
    const auto old_manifest = slurp(only_manifest(ns));
    const auto changed_private = b;
    for (const char* step : {"write", "file_flush", "rename"}) {
        SessionWriteOptions options;
        options.fault = [step](const char* at) { return std::strcmp(step, at) ? 0 : EIO; };
        check(!store.write("a", changed_private, {1}, bytes_b, error, options, &status) && !status.published &&
              status.error == SessionError::io && slurp(only_manifest(ns)) == old_manifest,
              "failed replacement preserves old committed manifest");
        check(store.read("a", restored, read_bytes, error) && same(a, restored), "old snapshot restorable after failed replacement");
    }
    auto changed_chunk = a;
    changed_chunk.kv[0].k.visit(0, 1, [](uint8_t* p, size_t, size_t) { *p ^= 128; return true; });
    SessionWriteOptions unpublished;
    int renames = 0;
    unpublished.fault = [&](const char* at) { return std::strcmp(at, "rename") == 0 && ++renames == 2 ? EIO : 0; };
    check(!store.write("a", changed_chunk, {1}, bytes_b, error, unpublished, &status) && !status.published &&
          renames == 2 && slurp(only_manifest(ns)) == old_manifest &&
          store.read("a", restored, read_bytes, error) && same(a, restored),
          "new block committed before failed manifest publish leaves old snapshot intact");
    check(store.entries(entries, stats, error) && stats.blocks == 9 && stats.physical_bytes == file_bytes(ns),
          "physical statistics include unpublished orphan chunk");
    const uint64_t committed_budget = stats.logical_bytes;
    check(store.prune(committed_budget, 1, stats, error) && stats.entries == 1 && stats.blocks == 8 &&
          stats.physical_bytes == file_bytes(ns) && store.read("a", restored, read_bytes, error) && same(a, restored),
          "orphan collection precedes physical-budget eviction");
    SessionWriteOptions post_publish;
    int directory_flushes = 0;
    post_publish.fault = [&](const char* at) {
        return std::strcmp(at, "dir_flush") == 0 && ++directory_flushes == 2 ? EIO : 0;
    };
    check(!store.write("a", changed_chunk, {1}, bytes_b, error, post_publish, &status) && status.published,
          "post-publication directory-flush failure truthfully reports committed replacement");
    check(store.prune(UINT64_MAX, SIZE_MAX, stats, error) && stats.entries == 1 && stats.blocks == 8 &&
          store.read("a", restored, read_bytes, error) && same(changed_chunk, restored),
          "immediate prune after failed manifest flush preserves newly referenced chunks");
    SessionBlockStore retry((root / "retry").string(), id);
    check(retry.write("a", a, {1}, bytes_a, error), "orphan retry fixture");
    SessionWriteOptions failed_block_flush;
    failed_block_flush.fault = [](const char* at) { return std::strcmp(at, "dir_flush") == 0 ? EIO : 0; };
    check(!retry.write("a", changed_chunk, {1}, bytes_b, error, failed_block_flush, &status) && !status.published,
          "failed new block flush does not publish a manifest");
    check(retry.write("a", changed_chunk, {1}, bytes_b, error) &&
          retry.read("a", restored, read_bytes, error) && same(changed_chunk, restored) &&
          retry.prune(UINT64_MAX, SIZE_MAX, stats, error) && stats.blocks == 8,
          "immediate retry reuses orphan candidate before complete manifest publication");
    check(store.write("a", a, {1}, bytes_a, error), "restore preflight fixture");
    check(store.prune(UINT64_MAX, SIZE_MAX, stats, error) && stats.blocks == 8, "restore clean preflight fixture");
    SessionWriteOptions no_space;
    no_space.min_free_bytes = UINT64_MAX;
    check(!store.write("a", b, {1}, bytes_b, error, no_space, &status) && status.error == SessionError::storage &&
          !status.published && slurp(only_manifest(ns)) == old_manifest, "free-space refusal preserves committed replacement");
    const fs::path committed_manifest = only_manifest(ns);
    auto bad_manifest = old_manifest;
    bad_manifest[64 + 24] ^= 1; // checked metadata, not the independently validated private body
    spit(committed_manifest, bad_manifest);
    const uint64_t before_corrupt_prune = file_bytes(ns);
    check(!store.prune(0, 0, stats, error, {}, &status) && status.error == SessionError::invalid &&
          file_bytes(ns) == before_corrupt_prune && slurp(committed_manifest) == bad_manifest,
          "corrupt committed owner prevents all reclamation");
    spit(committed_manifest, old_manifest);
    SessionReadLimits limits;
    limits.max_tokens = 4096;
    restored = b;
    check(!store.read("a", restored, read_bytes, error, limits, &status) && status.error == SessionError::invalid && same(b, restored),
          "bounded parser rejects before output replacement");
    limits = {}; limits.max_kv_bytes.resize(2); limits.max_kv_bytes[0].fill(1); limits.max_kv_bytes[1].fill(UINT64_MAX);
    check(!store.read("a", restored, read_bytes, error, limits, &status), "external chunk sizes obey per-part limits");
    limits = {}; limits.admit = [](uint64_t, std::string& why) { why = "probe admission"; return false; };
    check(!store.read("a", restored, read_bytes, error, limits, &status) && status.error == SessionError::memory,
          "logical image RAM admission before materialization");
    check(store.entries(entries, stats, error, limits), "metadata discovery does not request full image RAM");

    const fs::path block = fs::directory_iterator(ns / "blocks")->path();
    const auto valid_block = slurp(block);
    auto corrupt = valid_block; corrupt[64] ^= 1; spit(block, corrupt);
    check(!store.read("a", restored, read_bytes, error, {}, &status) && status.error == SessionError::invalid && same(b, restored),
          "corrupt chunk refused without partial output");
    check(store.write("new", a, {1}, bytes_b, error, {}, &status) && status.published &&
          store.erase("new", error) && slurp(only_manifest(ns)) == old_manifest,
          "write reuse accepts managed framing without auditing the referenced payload");
    rehash_envelope(corrupt); spit(block, corrupt);
    check(!store.read("a", restored, read_bytes, error, {}, &status) && status.error == SessionError::invalid,
          "valid envelope with mismatched actual-byte content key refused");
    check(!store.write("new", a, {1}, bytes_b, error, {}, &status) &&
          status.error == SessionError::invalid && !status.published,
          "write reuse refuses trailer content-key record inconsistent with managed filename");
    spit(block, valid_block);
    check(store.read("a", restored, read_bytes, error) && same(a, restored), "restored valid chunk fixture");

    SessionBlockStore other((root / "cache").string(), {id.model, id.config + 1});
    check(other.entries(entries, stats, error) && stats.entries == 0 && other.namespace_directory() != store.namespace_directory(),
          "model/config namespace separation");
    check(other.write("a", b, {1}, bytes_b, error), "other identity fixture");
    const fs::path other_manifest = only_manifest(fs::path(other.namespace_directory()));
    const auto other_saved = slurp(other_manifest);
    spit(other_manifest, old_manifest);
    check(!other.read("a", restored, read_bytes, error, {}, &status) && status.error == SessionError::invalid,
          "foreign identity manifest refused even inside namespace");
    spit(other_manifest, other_saved);

    check(store.write("b", b, {1}, bytes_b, error), "LRU second fixture");
    const auto old_time = fs::file_time_type::clock::now() - std::chrono::seconds(10);
    for (const auto& file : fs::directory_iterator(ns / "manifests")) fs::last_write_time(file.path(), old_time);
    check(store.touch("a", error), "successful restore touch");
    check(store.prune(UINT64_MAX, 1, stats, error) && stats.entries == 1 && stats.blocks == 8 &&
          store.read("a", restored, read_bytes, error) && same(a, restored), "touch ordering prunes older branch not recently restored entry");
    check(!store.read("b", restored, read_bytes, error), "older branch pruned");
    check(store.prune(0, 1, stats, error) && stats.entries == 0 && stats.blocks == 0 && file_bytes(ns) == 0,
          "physical budget prune releases shared chunks only at last owner");
    check(other.read("a", restored, read_bytes, error) && same(b, restored), "prune never touches sibling namespace");
    const fs::path unrelated = ns / "blocks" / "user.bin";
    spit(unrelated, {'k', 'e', 'e', 'p'});
    check(store.prune(0, 0, stats, error) && slurp(unrelated) == std::vector<char>({'k', 'e', 'e', 'p'}),
          "collector preserves unrelated files");
    auto no_draft = a; no_draft.kv.pop_back();
    check(store.write("no-draft", no_draft, {1}, bytes_a, error) && store.entries(entries, stats, error) &&
          entries[0].kv_layers == 1 && stats.blocks == 8 && store.read("no-draft", restored, read_bytes, error) &&
          same(no_draft, restored), "explicit main layer count also handles snapshot without draft");
    SessionBlockStore metadata_store((root / "metadata").string(), id);
    auto large = a;
    large.live.gdn.assign(20u << 20, 0x5a);
    large.checkpoints[0].ple.assign(5u << 20, 0x3c);
    check(metadata_store.write("large", large, {1}, bytes_a, error), "large private payload fixture");
    const fs::path metadata_ns(metadata_store.namespace_directory()), metadata_file = only_manifest(metadata_ns);
    auto private_corrupt = slurp(metadata_file);
    uint64_t metadata_length = 0;
    std::memcpy(&metadata_length, private_corrupt.data() + 40, 8);
    uint64_t bytes_read = 0, progress_events = 0;
    SessionReadLimits metadata_limits;
    metadata_limits.progress = [&](uint64_t done, uint64_t) { ++progress_events; bytes_read = std::max(bytes_read, done); };
    check(metadata_store.entries(entries, stats, error, metadata_limits) && entries.size() == 1 && stats.blocks == 8 &&
          progress_events >= 2 && bytes_read == 64 + metadata_length && bytes_read < large.live.gdn.size() / 100,
          "discovery performs positive bounded metadata IO without reading large private payload");
    std::printf("metadata discovery read %llu bytes of %llu-byte manifest\n",
                (unsigned long long) bytes_read, (unsigned long long) private_corrupt.size());
    check(metadata_store.read("large", restored, read_bytes, error) && same(large, restored),
          "full reader reconstructs independently framed large private payload");
    private_corrupt[(size_t) (64 + metadata_length)] ^= 1;
    spit(metadata_file, private_corrupt);
    restored = b;
    check(metadata_store.entries(entries, stats, error) && stats.entries == 1 && stats.blocks == 8 &&
          !metadata_store.read("large", restored, read_bytes, error, {}, &status) &&
          status.error == SessionError::invalid && error.find("private payload checksum") != std::string::npos && same(b, restored),
          "private corruption retains checked ownership but full read refuses without replacing output");
    check(metadata_store.prune(UINT64_MAX, SIZE_MAX, stats, error) && stats.entries == 1 && stats.blocks == 8,
          "valid metadata remains an owner despite private body corruption");
    auto metadata_corrupt = private_corrupt;
    metadata_corrupt[64 + 16] ^= 1;
    spit(metadata_file, metadata_corrupt);
    const uint64_t before_metadata_failure = file_bytes(metadata_ns);
    check(!metadata_store.entries(entries, stats, error, {}, &status) && status.error == SessionError::invalid &&
          !metadata_store.prune(0, 0, stats, error, {}, &status) && status.error == SessionError::invalid &&
          error.find("metadata checksum") != std::string::npos && file_bytes(metadata_ns) == before_metadata_failure,
          "checked metadata corruption fails discovery and all ownership-based reclamation closed");
    auto truncated = private_corrupt;
    truncated.resize((size_t) (64 + metadata_length + 1));
    spit(metadata_file, truncated);
    check(metadata_store.entries(entries, stats, error) && stats.blocks == 8 &&
          !metadata_store.read("large", restored, read_bytes, error) && same(b, restored),
          "truncated private body never loses valid metadata ownership");
    check(metadata_store.prune(0, 0, stats, error) && stats.blocks == 0 && stats.entries == 0,
          "checked directory permits eviction of unusable private payload without leaking chunks");
    SessionBlockStore gc_store((root / "framing-gc").string(), id);
    auto gc_image = a;
    gc_image.live.ids.resize(4096);
    ConversationKv gc_kv;
    gc_kv.cells = 4096; gc_kv.format = 2; gc_kv.heads = 1; gc_kv.head_dim = 2048; gc_kv.page_size = 64;
    gc_kv.k = pattern(4096, 2048, 35);
    gc_image.kv.clear(); gc_image.kv.push_back(std::move(gc_kv));
    check(gc_store.write("owner", gc_image, {1}, bytes_a, error), "single large managed chunk fixture");
    const fs::path gc_ns(gc_store.namespace_directory()), gc_block = fs::directory_iterator(gc_ns / "blocks")->path();
    const auto gc_valid = slurp(gc_block);
    auto gc_corrupt = gc_valid;
    gc_corrupt[64] ^= 1;
    spit(gc_block, gc_corrupt);
    restored = b;
    check(gc_store.write("owner", gc_image, {1}, bytes_a, error, {}, &status) && status.published,
          "replacement reuses frame-valid chunk without reading its payload");
    check(!gc_store.read("owner", restored, read_bytes, error, {}, &status) &&
          status.error == SessionError::invalid && same(b, restored),
          "restore of reused chunk rejects corrupt payload without replacing caller image");
    check(gc_store.prune(UINT64_MAX, SIZE_MAX, stats, error) && stats.entries == 1 && stats.blocks == 1 &&
          slurp(gc_block) == gc_corrupt, "referenced chunk survives GC even when payload is corrupt");
    check(gc_store.erase("owner", error) && !fs::exists(gc_block) &&
          gc_store.entries(entries, stats, error) && stats.entries == 0 && stats.blocks == 0,
          "last owner eviction reclaims header-identifiable chunk despite discarded payload corruption");
    const auto refuses_framing = [&](const std::vector<char>& forged, const char* label) {
        spit(gc_block, forged);
        check(!gc_store.write("owner", gc_image, {1}, bytes_a, error, {}, &status) &&
              status.error == SessionError::invalid && !status.published &&
              !gc_store.prune(0, 0, stats, error, {}, &status) && status.error == SessionError::invalid &&
              slurp(gc_block) == forged, label);
        fs::remove(gc_block);
    };
    auto forged = gc_valid;
    forged[0] ^= 1;
    refuses_framing(forged, "GC preserves unknown framing under a canonical managed filename");
    forged = gc_valid;
    forged[16] ^= 1;
    const uint64_t foreign_header_hash = session_hash64(forged.data(), 56, 0);
    std::memcpy(forged.data() + 56, &foreign_header_hash, 8);
    refuses_framing(forged, "GC preserves foreign identity with otherwise valid header checksum");
    forged = gc_valid;
    forged.pop_back();
    refuses_framing(forged, "GC refuses illegal managed chunk extent");
    forged = gc_valid;
    forged.back() ^= 1;
    refuses_framing(forged, "GC refuses invalid managed chunk trailer marker");
    forged = gc_valid;
    forged[forged.size() - 16] ^= 1;
    refuses_framing(forged, "GC refuses trailer content key inconsistent with managed filename");
#ifndef _WIN32
    const fs::path external = root / "gc-external-target";
    spit(external, gc_valid);
    fs::create_symlink(external, gc_block);
    check(!gc_store.prune(0, 0, stats, error, {}, &status) && status.error == SessionError::invalid &&
          fs::is_symlink(gc_block) && slurp(external) == gc_valid, "GC refuses symlink and preserves its target");
    fs::remove(gc_block);
    fs::create_hard_link(external, gc_block);
    check(!gc_store.prune(0, 0, stats, error, {}, &status) && status.error == SessionError::invalid &&
          fs::exists(gc_block) && fs::hard_link_count(external) == 2, "GC refuses hardlinked managed filename");
    fs::remove(gc_block);
#endif
    spit(gc_block, gc_corrupt);
    check(gc_store.prune(0, 0, stats, error) && stats.blocks == 0 && stats.entries == 0 && !fs::exists(gc_block),
          "orphan collector validates framing without needing discarded payload checksum");
    check(slurp(legacy / "manifests" / "legacy.session") == std::vector<char>({'k', 'e', 'e', 'p'}),
          "v2 writes and pruning preserve old v1 namespace files");
    {
        SessionBlockStore df2_store((root / "dflash2").string(), id);
        auto df2_a = a;
        df2_a.kv.resize(1);
        df2_a.dflash2.emplace();
        auto& draft = *df2_a.dflash2;
        draft.shape = {2560, 1, 32, 8, 128, 7680, 2048, 16, 256, 0, 32768,
                       2064, 2048, 256, 10000.0f, 1e-6f, false, {5}};
        draft.log_enabled = true;
        draft.epoch = 3;
        draft.have_ctx = draft.ring_hi = 8208;
        draft.ring.resize(2 * 8 * 2064 * 128, 0.0f);
        draft.ring[0] = 3.5f;
        draft.ring[8 * 2064 * 128] = -2.25f;
        draft.tags.assign(2064, -1);
        for (int64_t p = 6144; p < 8208; ++p) draft.tags[(size_t) (p % 2064)] = (int64_t(3) << 32) | p;
        draft.slot_positions = draft.tags;
        draft.history = df2_a.live.ids;
        draft.log.resize(1);
        auto& log = draft.log[0];
        log.index = 32;
        log.q.assign(256 * 2560, 4);
        log.scales.assign(256 * 80, 0.125f);
        log.valid.assign(256, 0);
        std::fill_n(log.valid.begin(), 16, uint8_t(1));
        draft.deferred_lo = 8208;
        draft.deferred_hi = 8209;
        draft.deferred.assign(2560, 0.25f);
        auto df2_b = b;
        df2_b.kv.resize(1);
        df2_b.live.ids.back() = 42;
        df2_b.dflash2 = draft;
        df2_b.dflash2->ring[0] = -13.0f;
        df2_b.dflash2->history = df2_b.live.ids;
        df2_b.dflash2->log[0].q[17] = -9;
        df2_b.dflash2->deferred[0] = 7.5f;
        check(df2_store.write("a", df2_a, {1}, bytes_a, error) &&
              df2_store.write("b", df2_b, {1}, bytes_b, error), "publish DFlash2 owners");
        const fs::path df2_ns(df2_store.namespace_directory());
        const auto df2_manifest = slurp(only_manifest(df2_ns));
        uint32_t df2_version = 0;
        std::memcpy(&df2_version, df2_manifest.data() + 8, sizeof df2_version);
        check(df2_version == 3, "DFlash2 manifests carry the versioned private-state extension");
        check(df2_store.entries(entries, stats, error) && stats.entries == 2 && stats.blocks == 8 &&
              stats.reused_bytes == expected_shared && stats.physical_bytes == file_bytes(df2_ns),
              "DFlash2 owners share only the eight main-model blocks");
        SessionReadLimits df2_limits;
        df2_limits.max_tokens = 8209;
        df2_limits.dflash2 = Dflash2SnapshotLimits{draft.shape, 8209, true};
        SessionBlockStore df2_reopened((root / "dflash2").string(), id);
        check(df2_reopened.read("a", restored, read_bytes, error, df2_limits) && same(df2_a, restored) &&
              restored.dflash2->ring[0] == 3.5f && restored.dflash2->ring[8 * 2064 * 128] == -2.25f &&
              restored.dflash2->tags[0] == ((int64_t(3) << 32) | 6192) &&
              restored.dflash2->log[0].q[17] == 4 && restored.dflash2->log[0].scales[0] == 0.125f &&
              restored.dflash2->history.back() == 8208 && restored.dflash2->deferred[0] == 0.25f,
              "reopen first owner with literal ring tags history log and deferred taps");
        check(df2_reopened.read("b", restored, read_bytes, error, df2_limits) && same(df2_b, restored) &&
              restored.dflash2->ring[0] == -13.0f && restored.dflash2->log[0].q[17] == -9 &&
              restored.dflash2->history.back() == 42 && restored.dflash2->deferred[0] == 7.5f,
              "second owner's DFlash2 state stays private");
        check(df2_store.erase("a", error) &&
              df2_reopened.read("b", restored, read_bytes, error, df2_limits) && same(df2_b, restored) &&
              df2_store.entries(entries, stats, error) && stats.entries == 1 && stats.blocks == 8,
              "remaining DFlash2 owner retains shared blocks and its own private state");
        check(df2_store.erase("b", error) && df2_store.entries(entries, stats, error) &&
              stats.entries == 0 && stats.blocks == 0 && file_bytes(df2_ns) == 0,
              "last DFlash2 owner releases the shared target blocks");
    }
    fs::remove_all(root);
    std::printf("conversation_blocks_test: %d checks passed\n", checks);
}
