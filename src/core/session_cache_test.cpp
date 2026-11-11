// Persisted asynchronous session-cache boundaries, including engine-thread-only capture-owner lifetime.
#include "strata/core/session_cache.hpp"
#include "strata/platform/aux_cpus.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

using namespace strata::core;
namespace fs = std::filesystem;
namespace {
int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, released = false;
#if defined(__linux__)
    cpu_set_t writer_mask{};
    bool have_writer_mask = false;
#endif
    int block(const char* step) {
        if (std::strcmp(step, "write")) return 0;
        std::unique_lock lock(mutex);
        if (entered) return 0;
#if defined(__linux__)
        CPU_ZERO(&writer_mask);
        have_writer_mask = sched_getaffinity(0, sizeof writer_mask, &writer_mask) == 0;
#endif
        entered = true; changed.notify_all();
        changed.wait(lock, [&] { return released; });
        return 0;
    }
    void wait() {
        std::unique_lock lock(mutex);
        check(changed.wait_for(lock, std::chrono::seconds(10), [&] { return entered; }), "writer reached production publisher gate");
    }
    void release() { std::lock_guard lock(mutex); released = true; changed.notify_all(); }
};
ConversationCheckpoint checkpoint(int n) {
    ConversationCheckpoint cp;
    for (int i = 0; i < n; ++i) cp.ids.push_back(i + 1);
    if (n > 3) cp.ids[3] = 248045;
    cp.imgs = {{2, 123}};
    cp.gdn = {1, 2, 3}; cp.ple = {4}; cp.tails = {5}; cp.dead = {6}; cp.block_pos = {7};
    return cp;
}
SavedConversation image(int branch = 0, bool draft = false) {
    SavedConversation out;
    out.geometry[0] = 42; out.layer_hi = 1; out.cvec = true;
    out.live = checkpoint(4101); out.live.ids.back() += branch;
    out.checkpoints = {checkpoint(4096)};
    out.live.ids[0] += branch * 100000;
    out.checkpoints[0].ids[0] = out.live.ids[0];
    ConversationKv kv;
    kv.cells = 8192; kv.heads = 1; kv.head_dim = 1; kv.format = 2; kv.page_size = 64;
    kv.k.resize(8192, 11); kv.v.resize(8192, 12);
    kv.k_scale.resize(8192, 13); kv.v_scale.resize(8192, 14);
    kv.pooled = {15, 16};
    out.kv.push_back(kv);
    if (draft) { kv.k = {21, 22, 23}; out.kv.push_back(kv); }
    return out;
}
std::vector<int64_t> prompt(const SavedConversation& snapshot) {
    std::vector<int64_t> ids(snapshot.live.ids.begin(), snapshot.live.ids.end());
    ids.push_back(99999);
    return ids;
}
SessionReadLimits limits() {
    SessionReadLimits out;
    out.geometry = image().geometry; out.layer_range = std::pair<int64_t, int64_t>{0, 1};
    out.max_tokens = 8192; out.max_checkpoints = 4; out.max_kv_layers = 2;
    return out;
}
std::vector<char> file_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
void replace_bytes(const fs::path& path, const std::vector<char>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), (std::streamsize) bytes.size());
}
DiskSessionCache::RequestPrefix request(int branch = 0) {
    return {{1 + branch * 100000, 2, 3}, {{2, 123}}, true};
}
void writer_placement(const fs::path& root, SessionFileIdentity identity) {
#if defined(__linux__)
    cpu_set_t original;
    CPU_ZERO(&original);
    check(sched_getaffinity(0, sizeof original, &original) == 0, "read test thread's original affinity");
    std::vector<int> allowed;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &original)) allowed.push_back(cpu);
    if (allowed.size() < 2) {
        std::printf("session_cache_test: writer affinity skipped (one allowed CPU)\n");
        return;
    }
    const int host = allowed.front(), auxiliary = allowed[1];
    std::array<bool, 2> full_ok{};
    std::string error;
    for (size_t mode = 0; mode < full_ok.size(); ++mode) {
        // configure must see the whole allowed mask before this host is pinned: exactly the engine's startup order.
        check(strata::aux_cpus::configure(mode ? std::to_string(auxiliary) : "off", error), "configure writer CPU fixture");
        (void) strata::aux_cpus::set_topology(host, {});
        cpu_set_t host_mask, expected = original;
        CPU_ZERO(&host_mask); CPU_SET(host, &host_mask);
        CPU_CLR(host, &expected);
        if (mode) { CPU_ZERO(&expected); CPU_SET(auxiliary, &expected); }
        check(sched_setaffinity(0, sizeof host_mask, &host_mask) == 0, "pin test host before creating real cache writers");
        {
            Gate gate;
            DiskSessionCache cache((root / ("writer-full-" + std::to_string(mode))).string(), 1 << 22, 2, identity, limits());
            SessionWriteOptions options;
            options.fault = [&](const char* step) { return gate.block(step); };
            cache.submit(request(), image(), {1}, options);
            gate.wait();
            full_ok[mode] = gate.have_writer_mask && !CPU_ISSET(host, &gate.writer_mask) &&
                            CPU_EQUAL(&expected, &gate.writer_mask);
            gate.release();
            check(cache.flush(error), "real full writer finishes affinity fixture");
        }
        strata::aux_cpus::set_enabled(false);
        check(sched_setaffinity(0, sizeof original, &original) == 0, "restore test host's original affinity");
    }
    check(strata::aux_cpus::configure("off", error), "restore default auxiliary configuration");
    (void) strata::aux_cpus::set_topology(-1, {});
    std::printf("session_cache_test: writer affinity full(off=%d aux=%d)\n",
                full_ok[0], full_ok[1]);
    check(full_ok[0], "full writer escapes inherited host affinity with auxiliary placement off");
    check(full_ok[1], "full writer respects the configured auxiliary CPU plan");
#else
    (void) root; (void) identity;
    std::printf("session_cache_test: writer affinity is Linux only\n");
#endif
}
}
int main() {
    const fs::path root = fs::temp_directory_path() / ("strata-disk-cache-test-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const SessionFileIdentity identity{123, 456};
    std::string error;
    const auto sample = image();
    {
        const auto rendered = DiskSessionCache::request_prefix({1, 2, 3, 248045, 77}, {{2, 123}}, true, 248045);
        check(rendered == request() &&
              DiskSessionCache::request_prefix({1, 2, 3}, {{2, 123}}, true, 248045) == request(),
              "logical identity is the exact input before the last assistant start, or the whole input without one");
        DiskSessionCache cache((root / "sessions").string(), 1 << 22, 2, identity, limits());
        auto admission = image(0, true); admission.live.ids.resize(4097);
        auto ram_admission = std::make_shared<const SavedConversation>(admission);
        cache.submit(request(), sample, {1}, {}, ram_admission);
        check(cache.flush(error), "full and admission companion publication");
        DiskSessionCache::Stats stats;
        check(cache.take_stats(stats) && stats.storage.entries == 1 && stats.saves == 1 && stats.storage.blocks == 4,
              "companion counts as one logical entry and private draft is not a shared block");
        const auto memory = DiskSessionCache::match(*ram_admission, prompt(sample), sample.live.imgs, true, 2);
        check(memory.tokens == 4097 && cache.take_stats(stats) == false && stats.reads == 0,
              "published immutable admission still supplies the real RAM resume point without a file read");
        auto full = cache.best(prompt(sample), sample.live.imgs, true, 1);
        auto solo = cache.best(prompt(sample), sample.live.imgs, true, 2);
        check(full.tokens == 4101 && solo.tokens == 4097 && full.key != solo.key, "batch uses completed image, solo uses exact draft-bearing admission");
        SavedConversation restored;
        size_t bytes = 0;
        check(cache.read(full, restored, bytes, error) && restored.live.ids == sample.live.ids && restored.kv.size() == 1 &&
              restored.kv[0].pooled == sample.kv[0].pooled, "production read preserves full private running and indexer state");
        check(cache.read(solo, restored, bytes, error) && restored.kv.size() == 2 && restored.kv.back().k == admission.kv.back().k,
              "admission restores its own exact private draft bytes");
        auto b = image(1);
        cache.submit(request(1), b, {1}); check(cache.flush(error), "second logical conversation publish");
        check(cache.take_stats(stats) && stats.storage.entries == 2 && stats.storage.blocks == 4 && stats.storage.reused_bytes > 0,
              "physical immutable blocks reused across native sessions");
        cache.submit(request(1), b, {1}); check(cache.flush(error), "duplicate save replaces logical key");
        check(cache.take_stats(stats) && stats.storage.entries == 2, "duplicate save does not consume extra logical capacity");
        // Force a deterministic age separation without sleeps or relying on filesystem timestamp resolution.
        SessionBlockStore files((root / "sessions" / "request-prefix-v1").string(), identity);
        for (const auto& file : fs::directory_iterator(fs::path(files.namespace_directory()) / "manifests"))
            fs::last_write_time(file.path(), fs::file_time_type::clock::now() - std::chrono::hours(1));
        check(cache.touch(full, error), "restore touches whole companion group");
        auto c = image(2); cache.submit(request(2), c, {1}); check(cache.flush(error), "third publish prunes LRU logical group");
        check(cache.take_stats(stats) && stats.storage.entries == 2 && stats.evictions == 1 &&
              cache.best(prompt(sample), sample.live.imgs, true, 2).tokens == 4097,
              "entry limit preserves companion and completion together");
        SessionWriteOptions failing;
        failing.fault = [](const char* step) { return std::strcmp(step, "rename") ? 0 : EIO; };
        auto replacement = sample; replacement.live.gdn = {99};
        cache.submit(request(), std::move(replacement), {1}, failing);
        check(!cache.flush(error) && !error.empty(), "full asynchronous publisher failure reaches FLUSH");
        check(cache.read(full, restored, bytes, error) && restored.live.gdn == sample.live.gdn,
              "failed asynchronous replacement leaves old committed full snapshot usable");
        check(cache.take_stats(stats) && stats.failures == 1 && stats.storage.entries == 2 && stats.writable,
              "failed write reports real failure and preserves valid cache budget metadata");
    }
    {
        DiskSessionCache cache((root / "sessions").string(), 1 << 22, 2, identity, limits());
        check(cache.best(prompt(sample), sample.live.imgs, true, 1).tokens == 4101 &&
              cache.best(prompt(sample), sample.live.imgs, true, 2).tokens == 4097,
              "native restart discovers full and admission manifests without Python index");
    }
    {
        const auto directory = root / "identity";
        SessionBlockStore legacy(directory.string(), identity);
        size_t bytes = 0;
        check(legacy.write("old-generated-key", sample, {1}, bytes, error), "legacy namespace fixture");
        DiskSessionCache cache(directory.string(), 1 << 22, 8, identity, limits());
        const auto prefix = request();
        auto draft = std::make_shared<const SavedConversation>(image(0, true));
        cache.submit(prefix, sample, {1}, {}, draft);
        check(cache.flush(error), "first exact input committed");
        auto changed = sample;
        changed.live.ids.push_back(248045); changed.live.ids.push_back(900);
        changed.live.gdn = {41, 42};
        changed.kv[0].k.visit(4097, 1, [](uint8_t* p, size_t, size_t) { *p = 127; return true; });
        cache.submit(prefix, changed, {1}, {}, draft);
        check(cache.flush(error), "same input with different output length and generated assistant marker committed");
        DiskSessionCache::Stats stats;
        check(cache.take_stats(stats) && stats.storage.entries == 1 && stats.saves == 2,
              "generation cap or generated suffix cannot create another logical entry");
        auto selected = cache.best(prompt(changed), changed.live.imgs, true, 1);
        SavedConversation restored;
        uint8_t tail = 0;
        check(cache.read(selected, restored, bytes, error) && restored.live.ids.size() == 4103 &&
              restored.live.ids.back() == 900 && restored.live.gdn == std::vector<uint8_t>({41, 42}) &&
              restored.kv[0].k.read(&tail, 4097, 1) && tail == 127,
              "same-key replacement contains the latest exact completed private bytes");
        auto with_draft = changed;
        with_draft.kv.push_back(draft->kv.back());
        with_draft.kv.back().k = {61, 62, 63};
        cache.submit(prefix, with_draft, {1}, {}, draft);
        check(cache.flush(error) && cache.take_stats(stats) && stats.storage.entries == 1,
              "main and slot draft/no-draft representations retain one logical identity");
        selected = cache.best(prompt(with_draft), with_draft.live.imgs, true, 2);
        check(selected.key.ends_with("-main") && cache.read(selected, restored, bytes, error) &&
              restored.kv.back().k == ConversationBuffer({61, 62, 63}) &&
              cache.best(prompt(with_draft), with_draft.live.imgs, true, 1).tokens == 0,
              "completed main member truthfully changes shape rather than retaining a stale completion");
        auto sibling = sample;
        sibling.live.ids[2] = sibling.checkpoints[0].ids[2] = 9;
        DiskSessionCache::RequestPrefix sibling_prefix{{1, 2, 9}, {{2, 123}}, true};
        cache.submit(sibling_prefix, sibling, {1});
        check(cache.flush(error) && cache.take_stats(stats) && stats.storage.entries == 2,
              "different requested prefixes sharing a system root remain separate");
        auto continuation = sample;
        continuation.live.ids[3] = continuation.checkpoints[0].ids[3] = 4;
        continuation.live.ids[4] = continuation.checkpoints[0].ids[4] = 248045;
        continuation.live.gdn = {81};
        DiskSessionCache::RequestPrefix continuation_prefix{{1, 2, 3, 4}, {{2, 123}}, true};
        auto inherited = image(0, true);
        inherited.live.ids[2] = inherited.checkpoints[0].ids[2] = 9;
        auto root_checkpoint = checkpoint(2); root_checkpoint.imgs.clear();
        inherited.checkpoints.push_back(root_checkpoint);
        auto inherited_owner = std::make_shared<const SavedConversation>(std::move(inherited));
        check(DiskSessionCache::match(*inherited_owner, prompt(continuation), continuation.live.imgs, true, 2).tokens == 2,
              "inherited draft is genuinely usable only before its different branch");
        cache.submit(continuation_prefix, continuation, {1}, {}, inherited_owner);
        check(cache.flush(error) && cache.take_stats(stats) && stats.storage.entries == 2,
              "requested-prefix continuation supersedes its predecessor but not the sibling");
        auto sibling_continuation = sibling;
        sibling_continuation.live.ids[4] = sibling_continuation.checkpoints[0].ids[4] = 8;
        sibling_continuation.live.ids[5] = sibling_continuation.checkpoints[0].ids[5] = 248045;
        DiskSessionCache::RequestPrefix sibling_continuation_prefix{{1, 2, 9, 248045, 8}, {{2, 123}}, true};
        cache.submit(sibling_continuation_prefix, sibling_continuation, {1});
        check(cache.flush(error) && cache.take_stats(stats) && stats.storage.entries == 2 &&
              cache.read(cache.best(prompt(continuation), continuation.live.imgs, true, 1), restored, bytes, error) &&
              restored.live.gdn == std::vector<uint8_t>({81}),
              "a long inherited draft never supplies another group's logical prefix for supersession");
        auto unsteered = continuation; unsteered.cvec = false;
        auto unsteered_prefix = continuation_prefix; unsteered_prefix.cvec = false;
        cache.submit(unsteered_prefix, unsteered, {1});
        auto other_image = continuation;
        other_image.live.imgs = other_image.checkpoints[0].imgs = {{2, 999}};
        auto other_image_prefix = continuation_prefix; other_image_prefix.images = {{2, 999}};
        cache.submit(other_image_prefix, other_image, {1});
        check(cache.flush(error) && cache.take_stats(stats) && stats.storage.entries == 4,
              "image identity and steering mode are part of the exact request identity");
        check(legacy.read("old-generated-key", restored, bytes, error) && restored.live.ids == sample.live.ids,
              "new request key domain leaves legacy cache manifests intact");
    }
    {
        fs::path corrupt_manifest;
        std::vector<char> original_manifest;
        {
            DiskSessionCache cache((root / "corrupt").string(), 1 << 22, 2, identity, limits());
            cache.submit(request(), sample, {1}); check(cache.flush(error), "corruption fixture committed");
            SessionBlockStore files((root / "corrupt" / "request-prefix-v1").string(), identity);
            const fs::path ns = fs::path(files.namespace_directory());
            const fs::path block = fs::directory_iterator(ns / "blocks")->path();
            const auto original_block = file_bytes(block);
            auto corrupt = original_block; corrupt[64] ^= 1; replace_bytes(block, corrupt);
            const auto match = cache.best(prompt(sample), sample.live.imgs, true, 1);
            SavedConversation retained = image(55);
            size_t bytes = 0;
            check(!cache.read(match, retained, bytes, error) && retained.live.ids.back() == 4156 &&
                  cache.best(prompt(sample), sample.live.imgs, true, 1).tokens == 0,
                  "corrupt automatic read preserves destination and refuses invalid key for this process");
            replace_bytes(block, original_block);
            corrupt_manifest = fs::directory_iterator(ns / "manifests")->path();
            original_manifest = file_bytes(corrupt_manifest);
            auto bad = original_manifest; bad[64 + 16] ^= 1; replace_bytes(corrupt_manifest, bad);
            cache.submit(request(2), image(2), {1});
            check(!cache.flush(error), "corrupt committed ownership makes async policy failure visible");
            DiskSessionCache::Stats stats;
            check(cache.take_stats(stats) && stats.failures >= 2 && !stats.writable && !stats.storage_known &&
                  file_bytes(corrupt_manifest) == bad, "failed ownership scan disables publication without unsafe reclamation");
            bool refused = false;
            try { cache.wait_for_room(); } catch (const std::runtime_error&) { refused = true; }
            check(refused, "budget failure stops further queued captures instead of growing cache without bound");
        }
        {
            DiskSessionCache cache((root / "corrupt").string(), 1 << 22, 2, identity, limits());
            DiskSessionCache::Stats stats;
            check(cache.take_stats(stats) && !stats.storage_known && !stats.writable && stats.failures == 1 &&
                  cache.best(prompt(sample), sample.live.imgs, true, 1).tokens == 0 &&
                  !cache.flush(error), "restart corruption stays explicit and permits caller cold inference");
        }
        replace_bytes(corrupt_manifest, original_manifest);
    }
    for (bool fail : {false, true}) {
        const auto owner = std::this_thread::get_id();
        std::atomic<bool> released{false}, owner_release{false};
        auto captured = image(0, true);
        captured.dflash2.emplace();
        auto& df2 = *captured.dflash2;
        df2.shape = {2560, 1, 32, 8, 128, 7680, 2048, 16, 256, 0, 32768,
                     2064, 2048, 256, 10000.0f, 1e-6f, false, {5}};
        constexpr size_t df2_ring_elements = 2 * 8 * 2064 * 128;
        df2.ring.assign(df2_ring_elements, 0.0f);
        df2.tags.assign(2064, -1);
        df2.slot_positions = df2.tags;
        df2.history = captured.live.ids;
        captured.pending = std::shared_ptr<void>(new uint8_t(0), [&](void* token) {
            owner_release = std::this_thread::get_id() == owner;
            delete static_cast<uint8_t*>(token);
            released = true;
        });
        Gate gate;
        DiskSessionCache cache((root / (fail ? "failed-owner" : "owner")).string(), 1 << 26, 2, identity, limits());
        SessionWriteOptions options;
        options.fault = [&](const char* step) {
            if (fail && std::strcmp(step, "rename") == 0) return EIO;
            return gate.block(step);
        };
        auto ram_admission = std::make_shared<const SavedConversation>(std::move(captured));
        cache.submit(request(), sample, {1}, options, ram_admission);
        ram_admission.reset(); // a slot can relinquish its RAM copy while publication is still in flight
        gate.wait();
        check(!released && cache.pending_bytes() >= sample.bytes() + df2_ring_elements * sizeof(float),
              "pending publication accounts for the immutable DFlash2 ring after caller releases it");
        check(cache.best(prompt(sample), sample.live.imgs, true, 2).tokens == 4101,
              "pending owned admission is a real resume candidate before publication");
        gate.release();
        check(cache.flush(error) == !fail && released && owner_release,
              "successful and failed publication retire final capture owner only on engine-thread collection");
    }
    writer_placement(root, identity);
    fs::remove_all(root);
    std::printf("session_cache_test OK (%d checks)\n", checks);
}
