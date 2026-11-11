// Automatic session cache: versioned manifests own immutable content-addressed 4096-cell K/V parts.
// Manifests check ownership metadata independently: v2 normally, v3 with private DFlash2 state.
// Existing cache namespaces and non-DFlash session-file v1 bytes are unchanged. One process owns a namespace.
#pragma once

#include "strata/core/conversation_file.hpp"
#include <mutex>

namespace strata::core {

struct SessionBlockLayout {
    // Explicit capture shape: [0, main_kv_layers) are main QSA layers; any remaining (draft) layers stay private.
    size_t main_kv_layers = 0;
};
struct SessionBlockCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    uint64_t used = 0;
};
struct SessionBlockEntry {
    std::string key;
    SessionBlockCheckpoint live;
    std::vector<SessionBlockCheckpoint> checkpoints;
    bool cvec = true;
    SessionBlockLayout layout;
    // stored = private manifest; shared = referenced chunk files; logical = their sum.
    // used = ordered filesystem publication/last successful touch time, independent of checkpoint stamps.
    uint64_t stored_bytes = 0, logical_bytes = 0, shared_bytes = 0, used = 0;
    size_t kv_layers = 0;
};
struct SessionBlockSummary {
    // Physical bytes/blocks include durable orphan chunks awaiting prune; logical/reused count committed owners.
    uint64_t entries = 0, blocks = 0, physical_bytes = 0, logical_bytes = 0, reused_bytes = 0;
};

class SessionBlockStore {
public:
    SessionBlockStore(std::string directory, SessionFileIdentity identity);
    // bytes = private manifest plus all referenced block files, not only newly written bytes.
    // Completed chunks in k/v/k_scale/v_scale are shared; partial tails, pooled/indexer, state and draft are private.
    bool write(const std::string& key, const SavedConversation& image, SessionBlockLayout layout,
               size_t& bytes, std::string& error, const SessionWriteOptions& options = {},
               SessionStatus* status = nullptr);
    bool read(const std::string& key, SavedConversation& image, size_t& bytes, std::string& error,
              const SessionReadLimits& limits = {}, SessionStatus* status = nullptr) const;
    // Reads only independently checked metadata, never private recurrent/KV bytes or their trailer.
    // A valid directory still owns chunks if private bytes corrupt; full read rejects that image.
    // Per-field/file bounds apply. Full-image RAM admission is only requested by read(), not discovery.
    bool entries(std::vector<SessionBlockEntry>& entries, SessionBlockSummary& summary, std::string& error,
                 const SessionReadLimits& limits = {}, SessionStatus* status = nullptr) const;
    // Removal/pruning reclaims only managed blocks unreferenced by every remaining committed manifest.
    bool erase(const std::string& key, std::string& error, SessionStatus* status = nullptr);
    // Call only after a successful validated restore; advances filesystem LRU without rereading private payloads.
    bool touch(const std::string& key, std::string& error, SessionStatus* status = nullptr);
    bool prune(uint64_t max_bytes, size_t max_entries, SessionBlockSummary& summary, std::string& error,
               const SessionReadLimits& limits = {}, SessionStatus* status = nullptr);
    std::string namespace_directory() const;
private:
    std::string directory_;
    SessionFileIdentity identity_;
    mutable std::mutex mutex_;
};

} // namespace strata::core
