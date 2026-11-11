#pragma once
#include "strata/core/conversation_blocks.hpp"
#include <array>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

namespace strata::core {
// One engine-thread owner. Workers only read captures; completed captures are destroyed by collect/drain on owner.
class DiskSessionCache {
public:
    struct RequestPrefix {
        std::vector<int32_t> ids;
        std::vector<ConversationImageKey> images;
        bool cvec = true;
        bool matches(const RequestPrefix& next) const {
            return cvec == next.cvec && images == next.images && ids.size() <= next.ids.size() &&
                   std::equal(ids.begin(), ids.end(), next.ids.begin());
        }
        bool operator==(const RequestPrefix&) const = default;
        size_t bytes() const {
            return ids.capacity() * sizeof(int32_t) + images.capacity() * sizeof(ConversationImageKey);
        }
    };
    using Admission = std::shared_ptr<const SavedConversation>;
    static RequestPrefix request_prefix(const std::vector<int64_t>&, const std::vector<ConversationImageKey>&,
                                        bool cvec, int64_t turn_token);
    struct Match { std::string key; int64_t tokens = 0; bool live = false; size_t kv_layers = 0; };
    struct Stats {
        bool storage_known = false, writable = true;
        SessionBlockSummary storage;
        uint64_t saves = 0, restores = 0, reads = 0, evictions = 0, failures = 0;
    };
    DiskSessionCache(std::string directory, uint64_t max_bytes, size_t max_entries,
                     SessionFileIdentity identity, SessionReadLimits limits);
    ~DiskSessionCache();
    void wait_for_room(); // call BEFORE capture: at most two write jobs, including the in-flight write
    void submit(RequestPrefix request, SavedConversation image, SessionBlockLayout layout,
                SessionWriteOptions options = {}, Admission admission = {});
    void collect();       // release completed capture owners on the engine thread only
    void drain();         // barrier without consuming errors; also releases completed leases
    bool flush(std::string& error);
    uint64_t pending_bytes() const;
    Match best(const std::vector<int64_t>& prompt, const std::vector<ConversationImageKey>& images,
               bool cvec, size_t kv_layers) const;
    static Match match(const SavedConversation&, const std::vector<int64_t>&,
                       const std::vector<ConversationImageKey>&, bool cvec, size_t kv_layers);
    void reject(const Match&, const std::string& error); // invalid before device transfer: refuse this key for the process
    bool read(const Match&, SavedConversation&, size_t& bytes, std::string& error);
    bool touch(const Match&, std::string& error);
    bool take_stats(Stats&); // facts changed since the last engine-thread publication
private:
    struct Pending {
        RequestPrefix request;
        Admission admission;
        std::string key;
        SavedConversation image;
        SessionBlockLayout layout;
        SessionWriteOptions options;
        std::string error;
        bool dir_flush_unsupported = false;
    };
    bool prune(std::vector<SessionBlockEntry>&, SessionBlockSummary&, uint64_t& evicted, std::string& error,
               const std::string& newest = {});
    static std::string key(const RequestPrefix&);
    void write_loop();
    SessionBlockStore store_;
    uint64_t max_bytes_;
    size_t max_entries_;
    SessionReadLimits limits_, read_limits_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::unique_ptr<Pending>> pending_;
    static constexpr size_t max_pending = 2;
    std::array<std::unique_ptr<Pending>, max_pending> retired_;
    size_t retired_count_ = 0; // fixed storage: completion and engine-thread collection cannot allocate
    std::vector<SessionBlockEntry> entries_;
    std::set<std::string> refused_;
    Stats stats_;
    bool stats_changed_ = true, stopping_ = false;
    std::string error_, write_refusal_;
    std::thread writer_;
};
} // namespace strata::core
