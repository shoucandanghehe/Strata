#include "strata/core/session_cache.hpp"
#include "strata/platform/aux_cpus.hpp"
#include <algorithm>
#include <charconv>
#include <filesystem>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <utility>

namespace strata::core {
namespace {
constexpr const char* main_suffix = "-main";
constexpr const char* draft_suffix = "-draft";
std::string request_directory(const std::string& directory) {
    const auto path = (std::filesystem::path(reinterpret_cast<const char8_t*>(directory.c_str())) /
                       "request-prefix-v1").u8string();
    return {path.begin(), path.end()};
}
std::string group_key(const std::string& key) {
    for (const char* suffix : {main_suffix, draft_suffix})
        if (key.ends_with(suffix)) return key.substr(0, key.size() - std::char_traits<char>::length(suffix));
    return key;
}
std::optional<size_t> request_extent(const std::string& key) {
    if (!key.starts_with("request-")) return {};
    const auto dash = key.find('-', 8);
    if (dash == std::string::npos) return {};
    size_t n = 0;
    const auto parsed = std::from_chars(key.data() + 8, key.data() + dash, n);
    return parsed.ec == std::errc{} && parsed.ptr == key.data() + dash ? std::optional<size_t>(n) : std::nullopt;
}
} // namespace
DiskSessionCache::DiskSessionCache(std::string directory, uint64_t max_bytes, size_t max_entries,
        SessionFileIdentity identity, SessionReadLimits limits)
    : store_(request_directory(directory), identity), max_bytes_(max_bytes), max_entries_(max_entries),
      limits_(limits), read_limits_(std::move(limits)) {
    // Metadata operations and the writer must never call a request's progress/admission closures.
    limits_.progress = {};
    limits_.admit = {};
    std::string error;
    uint64_t evicted = 0;
    if (!prune(entries_, stats_.storage, evicted, error)) {
        entries_.clear();
        error_ = write_refusal_ = error;
        ++stats_.failures;
        std::fprintf(stderr, "strata session cache: writes refused; inference remains cold: %s\n", error.c_str());
    }
    stats_.storage_known = write_refusal_.empty();
    writer_ = std::thread(&DiskSessionCache::write_loop, this);
}
DiskSessionCache::~DiskSessionCache() {
    std::string error;
    if (!flush(error)) std::fprintf(stderr, "strata session cache: shutdown flush failed: %s\n", error.c_str());
    { std::lock_guard lock(mutex_); stopping_ = true; }
    changed_.notify_all();
    writer_.join();
    collect();
}
DiskSessionCache::RequestPrefix DiskSessionCache::request_prefix(const std::vector<int64_t>& input,
        const std::vector<ConversationImageKey>& images, bool cvec, int64_t turn_token) {
    size_t boundary = input.size();
    for (size_t i = input.size(); i-- > 0;)
        if (input[i] == turn_token) { boundary = i; break; }
    return {{input.begin(), input.begin() + boundary}, images, cvec};
}
std::string DiskSessionCache::key(const RequestPrefix& request) {
    SessionIdentityBuilder hash(0);
    hash.bytes("tokens", request.ids.data(), request.ids.size() * sizeof(int32_t));
    hash.u64("cvec", request.cvec);
    for (const auto& img : request.images) { hash.i64("image.start", img.start); hash.u64("image.hash", img.hash); }
    char digest[17];
    std::snprintf(digest, sizeof digest, "%016llx", (unsigned long long) hash.digest());
    return "request-" + std::to_string(request.ids.size()) + "-" + digest;
}
void DiskSessionCache::collect() {
    std::array<std::unique_ptr<Pending>, max_pending> done;
    { std::lock_guard lock(mutex_); done.swap(retired_); retired_count_ = 0; }
    for (const auto& job : done) {
        if (!job) continue;
        if (!job->error.empty()) std::fprintf(stderr, "strata session cache: save failed: %s\n", job->error.c_str());
        else std::fprintf(stderr, "strata session cache: published %zu tokens\n", job->image.live.ids.size());
        if (job->dir_flush_unsupported)
            std::fprintf(stderr, "strata session cache: filesystem cannot flush the cache directory; new names are not flushed\n");
    }
    // `done` destroys every immutable capture (and its runtime owners) here, never in write_loop.
}
void DiskSessionCache::wait_for_room() {
    collect();
    {
        std::unique_lock lock(mutex_);
        if (!write_refusal_.empty()) throw std::runtime_error(write_refusal_);
        changed_.wait(lock, [&] { return pending_.size() < max_pending; });
        if (!write_refusal_.empty()) throw std::runtime_error(write_refusal_);
    }
    collect();
}
void DiskSessionCache::submit(RequestPrefix request, SavedConversation image, SessionBlockLayout layout,
                              SessionWriteOptions options, Admission admission) {
    wait_for_room();
    auto job = std::make_unique<Pending>();
    job->key = key(request);
    job->request = std::move(request);
    job->image = std::move(image);
    job->admission = std::move(admission);
    job->layout = layout;
    job->options = std::move(options);
    // Production callers use only min_free/durable. Never borrow command callbacks into a worker.
    job->options.progress = {};
    job->options.phase = {};
    {
        std::lock_guard lock(mutex_);
        if (!write_refusal_.empty()) throw std::runtime_error(write_refusal_);
        pending_.push_back(std::move(job));
    }
    changed_.notify_all();
}
void DiskSessionCache::drain() {
    { std::unique_lock lock(mutex_); changed_.wait(lock, [&] { return pending_.empty(); }); }
    collect();
}
bool DiskSessionCache::flush(std::string& error) {
    drain();
    std::lock_guard lock(mutex_);
    error = std::exchange(error_, {});
    return error.empty();
}
uint64_t DiskSessionCache::pending_bytes() const {
    std::lock_guard lock(mutex_);
    uint64_t bytes = 0;
    for (const auto& job : pending_) {
        auto count = [&](const SavedConversation& image) {
            for (const auto& layer : image.kv)
                bytes += layer.k.size() + layer.v.size() + layer.k_scale.size() + layer.v_scale.size() + layer.pooled.size();
            bytes += image.live.bytes();
            if (image.dflash2) bytes += image.dflash2->bytes();
            for (const auto& cp : image.checkpoints) bytes += cp.bytes();
        };
        count(job->image);
        if (job->admission) count(*job->admission);
    }
    return bytes;
}
bool DiskSessionCache::prune(std::vector<SessionBlockEntry>& entries, SessionBlockSummary& summary,
                             uint64_t& evicted, std::string& error, const std::string& newest) {
    if (!store_.prune(UINT64_MAX, SIZE_MAX, summary, error, limits_)) return false;
    if (!store_.entries(entries, summary, error, limits_)) return false;
    std::map<std::string, uint64_t> groups;
    for (const auto& entry : entries) groups[group_key(entry.key)] = std::max(groups[group_key(entry.key)], entry.used);
    while (!groups.empty() && (groups.size() > max_entries_ || summary.physical_bytes > max_bytes_)) {
        const auto oldest = std::min_element(groups.begin(), groups.end(), [&](const auto& a, const auto& b) {
            if ((a.first == newest) != (b.first == newest)) return b.first == newest;
            return a.second < b.second;
        });
        // The full image goes first. A crash can leave a valid admission prefix, never a dangling block reference.
        for (const std::string& key : {oldest->first + main_suffix, oldest->first + draft_suffix})
            if (std::any_of(entries.begin(), entries.end(), [&](const auto& entry) { return entry.key == key; }) &&
                !store_.erase(key, error)) return false;
        groups.erase(oldest);
        ++evicted;
        if (!store_.entries(entries, summary, error, limits_)) return false;
    }
    summary.entries = groups.size();
    return true;
}
void DiskSessionCache::write_loop() {
    (void) strata::aux_cpus::place_off_host();
    strata::aux_cpus::pin_current_thread();
    for (;;) {
        Pending* job;
        {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
            if (stopping_ && pending_.empty()) return;
            job = pending_.front().get();
        }
        size_t bytes = 0;
        SessionBlockSummary summary;
        std::vector<SessionBlockEntry> entries;
        bool saved = false;
        uint64_t evicted = 0;
        try {
            {
                std::lock_guard lock(mutex_);
                entries = entries_;
                if (!write_refusal_.empty()) throw std::runtime_error(write_refusal_);
            }
            SessionStatus status;
            bool admission_saved = true;
            if (job->admission) {
                admission_saved = store_.write(job->key + draft_suffix, *job->admission,
                    job->layout, bytes, job->error, job->options, &status);
                job->dir_flush_unsupported = status.dir_flush_unsupported;
            }
            if (admission_saved) {
                saved = store_.write(job->key + main_suffix, std::as_const(job->image), job->layout, bytes, job->error, job->options, &status);
                job->dir_flush_unsupported |= status.dir_flush_unsupported;
            }
            if (saved) {
                // Only a completed member owns this request's exact input. An inherited draft may belong
                // to another branch beyond its mounted checkpoint, even when its token count is longer.
                for (const auto& old : entries) {
                    if (!old.key.ends_with(main_suffix) || group_key(old.key) == job->key ||
                        old.cvec != job->request.cvec || old.live.imgs != job->request.images) continue;
                    const auto n = request_extent(old.key);
                    if (!n || *n > old.live.ids.size() || *n > job->request.ids.size() ||
                        !std::equal(old.live.ids.begin(), old.live.ids.begin() + *n, job->request.ids.begin())) continue;
                    const auto group = group_key(old.key);
                    if (!store_.erase(group + main_suffix, job->error) ||
                        !store_.erase(group + draft_suffix, job->error)) break;
                }
            }
        } catch (const std::exception& e) { job->error = e.what(); }
        bool consistent = false;
        std::string policy_error;
        try { consistent = prune(entries, summary, evicted, policy_error, saved ? job->key : std::string{}); }
        catch (const std::exception& e) { policy_error = e.what(); }
        if (!consistent && job->error.empty()) job->error = policy_error;
        {
            std::lock_guard lock(mutex_);
            if (!job->error.empty()) { if (error_.empty()) error_ = job->error; ++stats_.failures; }
            if (saved) ++stats_.saves;
            if (consistent) { entries_ = std::move(entries); stats_.storage = summary; stats_.evictions += evicted; }
            else write_refusal_ = policy_error;
            stats_.storage_known = consistent;
            stats_changed_ = true;
            // Unique ownership moves without destroying capture owners on this worker, even on failure.
            retired_[retired_count_++] = std::move(pending_.front());
            pending_.pop_front();
            job = nullptr;
        }
        changed_.notify_all();
    }
}
DiskSessionCache::Match DiskSessionCache::best(const std::vector<int64_t>& prompt,
        const std::vector<ConversationImageKey>& images, bool cvec, size_t kv_layers) const {
    Match match;
    std::lock_guard lock(mutex_);
    auto consider = [&](const std::string& key, const auto& cp, bool live) {
        const size_t n = cp.ids.size();
        if (refused_.contains(key) || (int64_t) n <= match.tokens || n >= prompt.size() ||
            !std::equal(cp.ids.begin(), cp.ids.end(), prompt.begin())) return;
        size_t j = 0;
        for (const auto& image : images) if (image.start < (int64_t) n) {
            if (j == cp.imgs.size() || cp.imgs[j++] != image) return;
        }
        if (j == cp.imgs.size()) match = {key, (int64_t) n, live, kv_layers};
    };
    for (const auto& entry : entries_) {
        if (entry.cvec != cvec || entry.kv_layers != kv_layers) continue;
        consider(entry.key, entry.live, true);
        for (const auto& cp : entry.checkpoints) consider(entry.key, cp, false);
    }
    // A queued immutable image is a candidate too. read() waits for its real publication rather than
    // claiming an early save or refetching live slot pointers to reconstruct its bytes.
    for (const auto& job : pending_) {
        auto pending = [&](const SavedConversation& image, const std::string& key) {
            if (image.cvec != cvec || image.kv.size() != kv_layers) return;
            consider(key, image.live, true);
            for (const auto& cp : image.checkpoints) consider(key, cp, false);
        };
        pending(job->image, job->key + main_suffix);
        if (job->admission) pending(*job->admission, job->key + draft_suffix);
    }
    return match;
}
DiskSessionCache::Match DiskSessionCache::match(const SavedConversation& image, const std::vector<int64_t>& prompt,
        const std::vector<ConversationImageKey>& images, bool cvec, size_t kv_layers) {
    Match best;
    if (image.cvec != cvec || image.kv.size() != kv_layers) return best;
    best.tokens = conversation_prefix(image.live, prompt, images);
    best.live = best.tokens > 0;
    best.kv_layers = kv_layers;
    for (const auto& cp : image.checkpoints) {
        const int64_t n = conversation_prefix(cp, prompt, images);
        if (n > best.tokens) { best.tokens = n; best.live = false; }
    }
    return best;
}
void DiskSessionCache::reject(const Match& match, const std::string& error) {
    std::lock_guard lock(mutex_);
    if (!refused_.insert(match.key).second) return;
    if (error_.empty()) error_ = error;
    ++stats_.failures;
    stats_changed_ = true;
}
bool DiskSessionCache::read(const Match& match, SavedConversation& image, size_t& bytes, std::string& error) {
    drain();
    { std::lock_guard lock(mutex_); ++stats_.reads; stats_changed_ = true; }
    SessionStatus status;
    if (store_.read(match.key, image, bytes, error, read_limits_, &status)) return true;
    std::lock_guard lock(mutex_);
    if (status.error == SessionError::invalid) refused_.insert(match.key);
    if (error_.empty()) error_ = error;
    ++stats_.failures;
    stats_changed_ = true;
    return false;
}
bool DiskSessionCache::touch(const Match& match, std::string& error) {
    drain();
    const std::string group = group_key(match.key);
    std::vector<std::string> keys{match.key};
    {
        std::lock_guard lock(mutex_);
        for (const auto& entry : entries_)
            if (entry.key != match.key && group_key(entry.key) == group) keys.push_back(entry.key);
    }
    for (const auto& key : keys) if (!store_.touch(key, error)) return false;
    std::lock_guard lock(mutex_);
    ++stats_.restores;
    stats_changed_ = true;
    return true;
}
bool DiskSessionCache::take_stats(Stats& stats) {
    std::lock_guard lock(mutex_);
    if (!stats_changed_) return false;
    stats = stats_;
    stats.writable = write_refusal_.empty();
    stats_changed_ = false;
    return true;
}
} // namespace strata::core
