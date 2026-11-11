#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_blocks.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <unordered_map>
#include <map>
#include <optional>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <malloc.h>
#include <process.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// Format v1 stores integers and floats as this machine holds them in memory; it is defined as little-endian with
// IEEE-754 floats, so a build where that is not the memory layout is refused rather than writing another format.
static_assert(std::endian::native == std::endian::little, "session files (format v1) are little-endian only");
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
              "session files (format v1) need IEEE-754 floats");

namespace strata::core {
namespace {
constexpr char kMagic[8] = {'S', 'T', 'R', 'S', 'E', 'S', 'S', '\x01'};
constexpr char kEnd[8] = {'S', 'T', 'R', 'S', 'E', 'N', 'D', '\x01'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeader = 64, kTrailer = 16;

// Progress for a transfer: one report after every block that was handed to (or read from) the OS, whatever its size
// - the last, partial block included - so a small file reports too, and each report follows real movement.
class Ticker {
public:
    explicit Ticker(const SessionProgress& fn) : fn_(fn) {}
    void operator()(uint64_t done, uint64_t total) { if (fn_) fn_(done, total); }
private:
    SessionProgress fn_;
};

SessionError errno_kind(int e) {
    switch (e) {
        case ENOSPC:
#ifdef EDQUOT
        case EDQUOT:
#endif
        case EFBIG: return SessionError::storage;
        case ENOMEM: return SessionError::memory;
        default: return SessionError::io;
    }
}

constexpr uint64_t P1 = 0x9E3779B185EBCA87ull, P2 = 0xC2B2AE3D27D4EB4Full, P3 = 0x165667B19E3779F9ull,
                   P4 = 0x85EBCA77C2B2AE63ull, P5 = 0x27D4EB2F165667C5ull;
uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
uint64_t round1(uint64_t acc, uint64_t w) { return rotl(acc + w * P2, 31) * P1; }
uint64_t load64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// ---- file I/O in aligned 16 MiB blocks.  Linux: O_DIRECT when the filesystem takes it (no page cache, so a
// 1.2 GB session neither evicts other files nor is charged to the engine's memory cgroup), buffered I/O otherwise
// (tmpfs, a filesystem that refuses an aligned direct transfer, or STRATA_SESSION_BUFFERED=1).  Windows and other
// systems: ordinary buffered I/O, same blocks.
constexpr size_t kAlign = 4096, kBlock = 16u << 20;
#ifdef _WIN32
struct AlignedFree { void operator()(uint8_t* p) const { _aligned_free(p); } };
using Aligned = std::unique_ptr<uint8_t, AlignedFree>;
Aligned aligned_block() { return Aligned(static_cast<uint8_t*>(_aligned_malloc(kBlock, kAlign))); }
#else
struct AlignedFree { void operator()(uint8_t* p) const { std::free(p); } };
using Aligned = std::unique_ptr<uint8_t, AlignedFree>;
Aligned aligned_block() { return Aligned(static_cast<uint8_t*>(std::aligned_alloc(kAlign, kBlock))); }
#endif

uint64_t random64() {
    std::random_device rd;
    uint64_t r = (uint64_t(rd()) << 32) ^ rd();
    r ^= (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() * P1;
#ifdef _WIN32
    r ^= (uint64_t) _getpid() * P2;
#else
    r ^= (uint64_t) ::getpid() * P2;
#endif
    return r;
}

// The temporary name beside the destination: hidden (a leading dot, which the server never accepts as a session
// name), unique, and short enough for any filesystem whatever the destination's length.
std::string temp_name(const std::string& leaf) {
    size_t keep = std::min<size_t>(leaf.size(), 64);
    while (keep > 0 && keep < leaf.size() && (uint8_t(leaf[keep]) & 0xC0) == 0x80) --keep;   // whole UTF-8 chars
    char hex[17];
    std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) random64());
    return "." + leaf.substr(0, keep) + "." + hex + ".tmp";
}

// a test's injected failure at `step`: its errno value (0 = none)
int injected(const std::function<int(const char*)>* fault, const char* step) {
    return fault && *fault ? (*fault)(step) : 0;
}

bool split_path(const std::string& path, std::string& dir, std::string& leaf, std::string& error) {
    const size_t cut = path.find_last_of(
#ifdef _WIN32
        "/\\"
#else
        "/"
#endif
    );
    dir = cut == std::string::npos ? std::string() : path.substr(0, cut == 0 ? 1 : cut);
    leaf = cut == std::string::npos ? path : path.substr(cut + 1);
    if (leaf.empty() || leaf == "." || leaf == "..") { error = "session file: " + path + " names no file"; return false; }
    return true;
}

#ifdef _WIN32
// paths reach the engine as UTF-8 (the server writes its commands as UTF-8); invalid UTF-8 is refused, not replaced
bool wide(const std::string& s, std::wstring& w) {
    w.clear();
    if (s.empty()) return true;
    if (s.size() > (size_t) INT_MAX) return false;
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), nullptr, 0);
    if (n <= 0) return false;
    w.assign((size_t) n, L'\0');
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int) s.size(), w.data(), n) == n;
}

SessionError win_kind(DWORD code) {
    switch (code) {
        case ERROR_DISK_FULL: case ERROR_HANDLE_DISK_FULL: case ERROR_FILE_TOO_LARGE: return SessionError::storage;
        case ERROR_NOT_ENOUGH_MEMORY: case ERROR_OUTOFMEMORY: return SessionError::memory;
        default: return SessionError::io;
    }
}

std::string last_error(DWORD code = GetLastError()) {
    char msg[256] = {};
    const DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, msg,
                                   sizeof msg, nullptr);
    std::string s(msg, n);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s.empty() ? "error " + std::to_string(code) : s;
}

class RawFile {
public:
    ~RawFile() { close(); }
    // a new file that did not exist (CREATE_NEW: never an existing file or link)
    bool create_new(const std::wstring& path) {
        h_ = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        return true;
    }
    // a session to restore: the name itself is opened, never what a link or reparse point there points to; it must be
    // an ordinary disk file with one name
    bool open_session(const std::string& path, uint64_t& size, bool for_flush = false) {
        std::wstring w;
        if (!wide(path, w)) { refused_ = true; error_ = "the path is not valid UTF-8"; return false; }
        h_ = CreateFileW(w.c_str(), GENERIC_READ | (for_flush ? GENERIC_WRITE : 0), FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                         FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        BY_HANDLE_FILE_INFORMATION info{};
        if (!GetFileInformationByHandle(h_, &info)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        if (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) { refused_ = true; error_ = "a link or reparse point, not a file"; return false; }
        if ((info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) || GetFileType(h_) != FILE_TYPE_DISK) {
            refused_ = true;
            error_ = "not a regular file";
            return false;
        }
        if (info.nNumberOfLinks != 1) { refused_ = true; error_ = "the file has more than one name (hard link)"; return false; }
        size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
        return true;
    }
    // a model input: links followed (model folders are often linked)
    bool open_model(const std::string& path, uint64_t& size) {
        std::wstring w;
        if (!wide(path, w)) { refused_ = true; error_ = "the path is not valid UTF-8"; return false; }
        h_ = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                         nullptr);
        if (h_ == INVALID_HANDLE_VALUE) {
            const DWORD code = GetLastError();
            missing_ = code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND;
            kind_ = win_kind(code); error_ = last_error(code);
            return false;
        }
        LARGE_INTEGER s{};
        if (!GetFileSizeEx(h_, &s)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        size = (uint64_t) s.QuadPart;
        return true;
    }
    bool is_open() const { return h_ != INVALID_HANDLE_VALUE; }
    bool direct() const { return false; }
    bool write_all(const uint8_t* p, size_t n) {
        while (n) {
            DWORD w = 0;
            const DWORD want = (DWORD) std::min<size_t>(n, size_t(1) << 30);
            if (!WriteFile(h_, p, want, &w, nullptr) || w == 0) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
            p += w; n -= w;
        }
        return true;
    }
    // bytes read, 0 at the end of the file, -1 on error
    long long read_some(uint8_t* p, size_t n) {
        DWORD r = 0;
        if (!ReadFile(h_, p, (DWORD) std::min<size_t>(n, size_t(1) << 30), &r, nullptr)) {
            { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
            return -1;
        }
        return (long long) r;
    }
    bool read_at(uint64_t offset, uint8_t* p, size_t n) {
        LARGE_INTEGER at{};
        at.QuadPart = (LONGLONG) offset;
        if (!SetFilePointerEx(h_, at, nullptr, FILE_BEGIN)) { { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); } return false; }
        while (n) {
            const long long r = read_some(p, n);
            if (r <= 0) { if (r == 0) error_ = "file ends early"; return false; }
            p += r; n -= (size_t) r;
        }
        return true;
    }
    bool truncate(uint64_t n) {
        LARGE_INTEGER at{};
        at.QuadPart = (LONGLONG) n;
        if (SetFilePointerEx(h_, at, nullptr, FILE_BEGIN) && SetEndOfFile(h_)) return true;
        { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
        return false;
    }
    bool sync() {
        if (FlushFileBuffers(h_)) return true;
        { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
        return false;
    }
    bool close() {
        if (h_ == INVALID_HANDLE_VALUE) return true;
        const bool ok = CloseHandle(h_) != 0;
        if (!ok) { const DWORD c_ = GetLastError(); kind_ = win_kind(c_); error_ = last_error(c_); }
        h_ = INVALID_HANDLE_VALUE;
        return ok;
    }
    const std::string& error() const { return error_; }
    bool missing() const { return missing_; }
    SessionError kind() const { return kind_; }
    // a validation refusal (a link, not a regular file, ...): bad input, not an I/O failure
    bool refused() const { return refused_; }
private:
    HANDLE h_ = INVALID_HANDLE_VALUE;
    SessionError kind_ = SessionError::io;
    bool refused_ = false;
    bool missing_ = false;
    std::string error_;
};

// Where the file is published.  MoveFileExW replaces an existing destination (std::rename does not on Windows); the
// replacement is a rename on one volume, but Windows promises no transaction on every filesystem.  A directory cannot
// be flushed on Windows: the file's data was flushed (FlushFileBuffers) before the move, and the move is write-through.
class Publisher {
public:
    bool open(const std::string& path, std::string& error) {
        std::string dir, leaf;
        if (!split_path(path, dir, leaf, error)) return false;
        std::wstring wdir, wleaf;
        if (!wide(dir, wdir) || !wide(leaf, wleaf)) { error = "session file: the path is not valid UTF-8"; return false; }
        prefix_ = wdir.empty() ? std::wstring() : (wdir.back() == L'\\' || wdir.back() == L'/' ? wdir : wdir + L"\\");
        final_ = prefix_ + wleaf;
        leaf_ = leaf;
        return true;
    }
    bool create_temp(RawFile& f, std::string& error) {
        for (int attempt = 0; attempt < 8; ++attempt) {
            std::wstring name;
            wide(temp_name(leaf_), name);
            temp_ = prefix_ + name;
            if (f.create_new(temp_)) { own_ = true; return true; }
            if (GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS) break;
        }
        error = f.error();
        return false;
    }
    bool publish(std::string& error, bool, SessionStatus& st, const std::function<int(const char*)>* fault) {
        if (const int e = injected(fault, "rename")) { error = std::strerror(e); st.error = errno_kind(e); return false; }
        if (MoveFileExW(temp_.c_str(), final_.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            own_ = false;
            st.published = true;
            // Windows has no folder flush; a test can still make this step fail after the move
            if (const int e = injected(fault, "dir_flush")) {
                error = std::string("flushing the directory: ") + std::strerror(e);
                st.error = errno_kind(e);
                return false;
            }
            return true;
        }
        const DWORD c = GetLastError();
        st.error = win_kind(c);
        error = last_error(c);
        return false;
    }
    bool sync_directory(std::string&, SessionStatus&) { return true; }
    bool erase(std::string& error, SessionStatus& st) {
        if (DeleteFileW(final_.c_str())) { st.published = true; return true; }
        const DWORD c = GetLastError();
        st.error = win_kind(c); error = last_error(c); return false;
    }
    SessionError kind() const { return SessionError::io; }
    ~Publisher() { if (own_) DeleteFileW(temp_.c_str()); }   // only the temporary file this call created
private:
    std::wstring prefix_, final_, temp_;
    std::string leaf_;
    bool own_ = false;
};

bool free_space(const std::string& path, uint64_t& out) {
    std::wstring w;
    if (!wide(session_free_space_dir(path, true), w)) return false;
    ULARGE_INTEGER avail{};
    if (!GetDiskFreeSpaceExW(w.empty() ? nullptr : w.c_str(), &avail, nullptr, nullptr)) return false;
    out = avail.QuadPart;
    return true;
}
#else
#ifndef O_DIRECT
#define O_DIRECT 0            // not offered by this system: buffered I/O
#endif

bool buffered_forced() {
    const char* e = std::getenv("STRATA_SESSION_BUFFERED");
    return e != nullptr && e[0] == '1';
}

static_assert(sizeof(off_t) >= 8, "session files need 64-bit file offsets (large-file support)");

class RawFile {
public:
    ~RawFile() { close(); }
    // a new file that did not exist (O_CREAT|O_EXCL: never an existing file or link), readable by its owner only
    bool create_new(int dirfd, const std::string& name) {
        fd_ = ::openat(dirfd, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (fd_ < 0) { kind_ = errno_kind(errno); error_ = std::strerror(errno); return false; }
        try_direct();
        return true;
    }
    // a session to restore: the name itself is opened, never what a symbolic link there points to; it must be a
    // regular file with one name (no hard link to a file elsewhere).  Checked on the opened descriptor.
    bool open_session(const std::string& path, uint64_t& size, bool for_flush = false) {
        fd_ = ::open(path.c_str(), (for_flush ? O_RDWR : O_RDONLY) | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
        if (fd_ < 0) {
            refused_ = errno == ELOOP;
            kind_ = errno_kind(errno);
            error_ = errno == ELOOP ? std::string("a symbolic link, not a file") : std::string(std::strerror(errno));
            return false;
        }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { error_ = std::strerror(errno); return false; }
        if (!S_ISREG(st.st_mode)) { refused_ = true; error_ = "not a regular file"; return false; }
        if (st.st_nlink != 1) { refused_ = true; error_ = "the file has more than one name (hard link)"; return false; }
        const int fl = ::fcntl(fd_, F_GETFL);
        if (fl >= 0) ::fcntl(fd_, F_SETFL, fl & ~O_NONBLOCK);
        size = (uint64_t) st.st_size;
        try_direct();
        return true;
    }
    // a model input: links followed (model folders are often linked)
    bool open_model(const std::string& path, uint64_t& size) {
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) { missing_ = errno == ENOENT; kind_ = errno_kind(errno); error_ = std::strerror(errno); return false; }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { error_ = std::strerror(errno); return false; }
        if (!S_ISREG(st.st_mode)) { error_ = "not a regular file"; return false; }
        size = (uint64_t) st.st_size;
        return true;
    }
    bool is_open() const { return fd_ >= 0; }
    bool direct() const { return direct_; }
    bool write_all(const uint8_t* p, size_t n) {
        while (n) {
            const ssize_t w = ::write(fd_, p, n);
            if (w < 0 && errno == EINTR) continue;
            // a filesystem that took O_DIRECT at open but refuses this transfer: nothing was written, go buffered
            if (w < 0 && errno == EINVAL && direct_ && leave_direct()) continue;
            if (w <= 0) { kind_ = w < 0 ? errno_kind(errno) : SessionError::io; error_ = w < 0 ? std::strerror(errno) : "short write"; return false; }
            p += w; n -= (size_t) w;
        }
        return true;
    }
    // bytes read, 0 at the end of the file, -1 on error
    long long read_some(uint8_t* p, size_t n) {
        for (;;) {
            const ssize_t r = ::read(fd_, p, n);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0 && errno == EINVAL && direct_ && leave_direct()) continue;
            if (r < 0) { kind_ = errno_kind(errno); error_ = std::strerror(errno); }
            return (long long) r;
        }
    }
    bool read_at(uint64_t offset, uint8_t* p, size_t n) {
        while (n) {
            const ssize_t r = ::pread(fd_, p, n, (off_t) offset);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0 && errno == EINVAL && direct_ && leave_direct()) continue;
            if (r <= 0) { kind_ = r < 0 ? errno_kind(errno) : SessionError::io; error_ = r < 0 ? std::strerror(errno) : "file ends early"; return false; }
            p += r; n -= (size_t) r; offset += (uint64_t) r;
        }
        return true;
    }
    bool truncate(uint64_t n) {
        if (::ftruncate(fd_, (off_t) n) == 0) return true;
        kind_ = errno_kind(errno);
        error_ = std::strerror(errno);
        return false;
    }
    bool sync() {
        if (::fsync(fd_) == 0) return true;
        kind_ = errno_kind(errno);
        error_ = std::strerror(errno);
        return false;
    }
    bool close() {
        if (fd_ < 0) return true;
        const bool ok = ::close(fd_) == 0;
        if (!ok) { kind_ = errno_kind(errno); error_ = std::strerror(errno); }
        fd_ = -1;
        return ok;
    }
    const std::string& error() const { return error_; }
    bool missing() const { return missing_; }
    SessionError kind() const { return kind_; }
    // a validation refusal (a link, not a regular file, ...): bad input, not an I/O failure
    bool refused() const { return refused_; }
private:
    // O_DIRECT is switched on the open descriptor: a filesystem that refuses it (EINVAL, e.g. tmpfs) keeps buffered
    // I/O, and the file is never opened twice
    void try_direct() {
        if (O_DIRECT == 0 || buffered_forced()) return;
        const int fl = ::fcntl(fd_, F_GETFL);
        direct_ = fl >= 0 && ::fcntl(fd_, F_SETFL, fl | O_DIRECT) == 0;
    }
    bool leave_direct() {
        const int fl = ::fcntl(fd_, F_GETFL);
        if (fl < 0 || ::fcntl(fd_, F_SETFL, fl & ~O_DIRECT) != 0) return false;
        direct_ = false;
        return true;
    }
    int fd_ = -1;
    bool direct_ = false, missing_ = false, refused_ = false;
    SessionError kind_ = SessionError::io;
    std::string error_;
};

// The destination directory is opened once; the temporary file is created in it exclusively, renamed over the
// final name in the same directory (a symbolic link at the final name is replaced, never written through), and the
// directory is flushed after the rename so the new name survives a power loss.
class Publisher {
public:
    bool open(const std::string& path, std::string& error) {
        std::string dir;
        if (!split_path(path, dir, leaf_, error)) return false;
        dirfd_ = ::open(dir.empty() ? "." : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (dirfd_ < 0) {
            kind_ = errno_kind(errno);
            error = "session file: cannot open the directory of " + path + ": " + std::strerror(errno);
            return false;
        }
        return true;
    }
    bool create_temp(RawFile& f, std::string& error) {
        for (int attempt = 0; attempt < 8; ++attempt) {
            temp_ = temp_name(leaf_);
            if (f.create_new(dirfd_, temp_)) { own_ = true; return true; }
            if (errno != EEXIST) break;
        }
        error = f.error();
        return false;
    }
    bool publish(std::string& error, bool durable, SessionStatus& st, const std::function<int(const char*)>* fault) {
        int e = injected(fault, "rename");
        if (e || ::renameat(dirfd_, temp_.c_str(), dirfd_, leaf_.c_str()) != 0) {
            if (!e) e = errno;
            error = std::strerror(e);
            st.error = errno_kind(e);
            return false;
        }
        own_ = false;
        st.published = true;   // the new file is at its name now: a failure below cannot bring the old one back
        if (!durable) return true;
        e = injected(fault, "dir_flush");
        if (!e && ::fsync(dirfd_) != 0) e = errno;
        if (e == EINVAL) { st.dir_flush_unsupported = true; return true; }   // this filesystem cannot flush a folder
        if (e) {
            error = std::string("flushing the directory: ") + std::strerror(e);
            st.error = errno_kind(e);
            return false;
        }
        return true;
    }
    // Flush the ownership removal before any block can be reclaimed.
    bool erase(std::string& error, SessionStatus& st) {
        if (::unlinkat(dirfd_, leaf_.c_str(), 0) != 0) {
            st.error = errno_kind(errno); error = std::strerror(errno); return false;
        }
        st.published = true;
        return sync_directory(error, st);
    }
    bool sync_directory(std::string& error, SessionStatus& st) {
        if (::fsync(dirfd_) == 0) return true;
        const int e = errno;
        if (e == EINVAL) { st.dir_flush_unsupported = true; return true; }
        st.error = errno_kind(e); error = std::strerror(e); return false;
    }
    SessionError kind() const { return kind_; }
    ~Publisher() {
        if (own_) ::unlinkat(dirfd_, temp_.c_str(), 0);   // only the temporary file this call created
        if (dirfd_ >= 0) ::close(dirfd_);
    }
private:
    int dirfd_ = -1;
    std::string leaf_, temp_;
    bool own_ = false;
    SessionError kind_ = SessionError::io;
};

bool free_space(const std::string& path, uint64_t& out) {
    struct statvfs s {};
    if (::statvfs(session_free_space_dir(path, false).c_str(), &s) != 0) return false;
    out = (uint64_t) s.f_bavail * (uint64_t) s.f_frsize;
    return true;
}
#endif

class FileSink {
public:
    explicit FileSink(RawFile& f) : buf_(aligned_block()), f_(f) {}
    bool ok() const { return buf_ != nullptr; }
    std::string error() const {
        if (!buf_) return "out of memory for the write buffer";
        return fault_ ? fault_msg_ : f_.error();
    }
    SessionError kind() const { return buf_ ? (fault_ ? fault_kind_ : f_.kind()) : SessionError::memory; }
    // room in the block for in-place production (avoids a staging copy)
    uint8_t* span(size_t& room) { room = kBlock - used_; return buf_.get() + used_; }
    bool commit(size_t n) { used_ += n; total_ += n; return used_ < kBlock || drain(); }
    bool write(const void* p, size_t n) {
        const uint8_t* s = static_cast<const uint8_t*>(p);
        while (n) {
            size_t room = 0;
            uint8_t* d = span(room);
            const size_t c = std::min(room, n);
            std::memcpy(d, s, c);
            if (!commit(c)) return false;
            s += c; n -= c;
        }
        return true;
    }
    // last block padded to the alignment for direct I/O, then the file cut back to its exact size and flushed
    bool finish(bool durable) {
        if (!f_.is_open()) return false;
        if (used_ && write_fault) fail_with(write_fault);
        if (fault_) { f_.close(); return false; }
        const size_t padded = f_.direct() ? (used_ + kAlign - 1) / kAlign * kAlign : used_;
        std::memset(buf_.get() + used_, 0, padded - used_);
        bool ok = f_.write_all(buf_.get(), padded);
        if (ok && used_ && progress) (*progress)(total_, expected);   // the last, partial block
        ok = ok && (padded == used_ || f_.truncate(total_));
        if (ok && durable && phase) phase("flush", total_);
        if (ok && durable && flush_fault) { fail_with(flush_fault); ok = false; }
        ok = ok && (!durable || f_.sync());
        ok = f_.close() && ok;
        return ok;
    }
    std::optional<Ticker> progress;
    SessionPhase phase;
    uint64_t expected = 0;
    int write_fault = 0, flush_fault = 0;   // tests: errno values injected at the first block write / the file flush
private:
    void fail_with(int e) { fault_ = true; fault_kind_ = errno_kind(e); fault_msg_ = std::strerror(e); }
    bool drain() {
        if (write_fault) { fail_with(write_fault); return false; }
        if (!f_.write_all(buf_.get(), used_)) return false;
        used_ = 0;
        if (progress) (*progress)(total_, expected);
        return true;
    }
    Aligned buf_;
    RawFile& f_;
    bool fault_ = false;
    SessionError fault_kind_ = SessionError::io;
    std::string fault_msg_;
    size_t used_ = 0;
    uint64_t total_ = 0;
};

class FileSource {
public:
    explicit FileSource(RawFile& f) : buf_(aligned_block()), f_(f) {}
    bool ok() const { return buf_ != nullptr; }
    std::string error() const { return buf_ ? f_.error() : std::string("out of memory for the read buffer"); }
    SessionError kind() const { return buf_ ? f_.kind() : SessionError::memory; }
    bool read(void* p, size_t n) {
        uint8_t* d = static_cast<uint8_t*>(p);
        while (n) {
            if (at_ == have_ && !fill()) return false;
            const size_t c = std::min(n, have_ - at_);
            std::memcpy(d, buf_.get() + at_, c);
            at_ += c; d += c; n -= c;
        }
        return true;
    }
    // Metadata-only readers cap OS transfers at their checked prefix; ordinary/full readers keep aligned streaming.
    void read_limit(uint64_t bytes) { read_limit_ = bytes; }
    std::optional<Ticker> progress;
    uint64_t expected = 0;
private:
    bool fill() {
        at_ = have_ = 0;
        const size_t want = (size_t) std::min<uint64_t>(kBlock, read_limit_ - total_);
        while (have_ < want) {
            const long long r = f_.read_some(buf_.get() + have_, want - have_);
            if (r < 0) return false;
            if (r == 0) break;
            have_ += (size_t) r;
            if (f_.direct() && have_ % kAlign) break;   // short aligned read: end of file
        }
        const uint64_t before = total_;
        total_ += have_;
        if (progress && total_ > before) (*progress)(total_, expected);
        return have_ > 0;
    }
    Aligned buf_;
    RawFile& f_;
    size_t at_ = 0, have_ = 0;
    uint64_t total_ = 0;
    uint64_t read_limit_ = UINT64_MAX;
};

// ---- serialization: the same walk counts (f == nullptr) and writes
struct Out {
    FileSink* f = nullptr;
    SessionHasher* hash = nullptr;
    uint64_t n = 0;
    bool ok = true, overflow = false;
    void count(uint64_t c) {
        if (c > UINT64_MAX - n) { overflow = true; ok = false; return; }
        n += c;
    }
    void raw(const void* p, size_t c) {
        count(c);
        if (!ok || !c) return;
        if (hash) hash->update(p, c);
        if (f) ok = f->write(p, c);
    }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    template<class T> void vec(const std::vector<T>& v) { u64(v.size()); raw(v.data(), v.size() * sizeof(T)); }
    void buffer(const ConversationBuffer& b) {
        u64(b.size());
        b.visit(0, b.size(), [&](const uint8_t* p, size_t c, size_t) { raw(p, c); return ok; });
    }
    // a K/V part pulled from its owner straight into the file block (sizing pass: counted, never read)
    void source(const SessionKvSource& s, size_t part) {
        const size_t size = s.sizes[part];
        u64(size);
        if (!f) { count(size); return; }
        for (size_t at = 0; at < size && ok;) {
            size_t room = 0;
            uint8_t* d = f->span(room);
            const size_t c = std::min(room, size - at);
            if (!s.read || !s.read(part, at, d, c)) {
                ok = false;
                read_failed = true;
                if (s.error) read_error = s.error();
                return;
            }
            if (hash) hash->update(d, c);
            count(c);
            ok = ok && f->commit(c);
            at += c;
        }
    }
    bool read_failed = false;
    std::string read_error;
};

void put_kv_header(Out& o, int format, int64_t cells, int64_t heads, int64_t head_dim, int64_t page_size,
                   int64_t pooled_rows, int64_t idx_dim) {
    for (int64_t v : {int64_t(format), cells, heads, head_dim, page_size, pooled_rows, idx_dim}) o.i64(v);
}

void put_checkpoint(Out& o, const ConversationCheckpoint& c) {
    o.vec(c.ids);
    o.u64(c.imgs.size());
    for (const auto& k : c.imgs) { o.i64(k.start); o.u64(k.hash); }
    o.vec(c.gdn); o.vec(c.ple); o.vec(c.tails); o.vec(c.dead); o.vec(c.block_pos);
    o.u64(c.used);
}

// v2's only extension: schema 1, complete DFlash shape, counters, ring, tag shadows, current lineage and deferred taps.
// Kept separate so the immutable-block payload can use the identical bounded codec.
void put_dflash2(Out& o, const Dflash2State& s) {
    o.u64(1);
    const auto& g = s.shape;
    for (int64_t v : {g.hidden, g.layers, g.heads, g.kv_heads, g.head_dim, g.inter, g.window, g.group, g.rank,
                      g.mask_token, g.vocab, g.ring_slots, g.deferred_slots, g.log_chunk_rows}) o.i64(v);
    o.raw(&g.theta, sizeof(float)); o.raw(&g.eps, sizeof(float));
    o.u64(g.q4 ? 1 : 0); o.vec(g.taps);
    o.u64(s.log_enabled ? 1 : 0);
    o.i64(s.epoch); o.i64(s.have_ctx); o.i64(s.ring_hi);
    o.i64(s.deferred_lo); o.i64(s.deferred_hi);
    o.vec(s.ring); o.vec(s.tags); o.vec(s.slot_positions); o.vec(s.history);
    o.u64(s.log.size());
    for (const auto& c : s.log) { o.u64(c.index); o.vec(c.q); o.vec(c.scales); o.vec(c.valid); }
    o.vec(s.deferred);
}
void put_payload(Out& o, const SavedConversation& s, const std::vector<SessionKvSource>* sources = nullptr) {
    for (int64_t g : s.geometry) o.i64(g);
    o.i64(s.layer_lo); o.i64(s.layer_hi); o.u64(s.cvec ? 1 : 0);
    put_checkpoint(o, s.live);
    o.u64(s.checkpoints.size());
    for (const auto& c : s.checkpoints) put_checkpoint(o, c);
    if (sources) {
        o.u64(sources->size());
        for (const auto& k : *sources) {
            put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
            for (size_t part = 0; part < 5; ++part) o.source(k, part);
        }
        if (s.dflash2) put_dflash2(o, *s.dflash2);
        return;
    }
    o.u64(s.kv.size());
    for (const auto& k : s.kv) {
        put_kv_header(o, k.format, k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim);
        o.buffer(k.k); o.buffer(k.v); o.buffer(k.k_scale); o.buffer(k.v_scale); o.buffer(k.pooled);
    }
    if (s.dflash2) put_dflash2(o, *s.dflash2);
}

// ---- parsing: every count is checked against the bytes left and the caller's limits before anything is allocated
struct In {
    FileSource* f;
    SessionHasher hash;
    uint64_t left;
    std::string& error;
    const SessionReadLimits& limits;
    SessionError kind = SessionError::invalid;
    size_t kv_layer = 0;
    bool fail(const std::string& m) { if (error.empty()) error = "session file: " + m; return false; }
    bool raw(void* p, size_t c) {
        if (c > left) return fail("payload ends early");
        if (c && !f->read(p, c)) { kind = f->kind(); return fail("read error: " + f->error()); }
        hash.update(p, c);
        left -= c;
        return true;
    }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    // `elem`: the fewest file bytes one element takes; `limit`: the most elements the caller accepts
    bool count(uint64_t& n, size_t elem, uint64_t limit = UINT64_MAX) {
        if (!u64(n)) return false;
        if (elem && n > left / elem) return fail("element count exceeds the payload");
        if (n > limit) return fail("element count " + std::to_string(n) + " exceeds this engine's limit " +
                                   std::to_string(limit));
        if (n > (uint64_t) std::numeric_limits<size_t>::max() / (elem ? elem : 1)) return fail("element count too large");
        return true;
    }
    template<class T> bool vec(std::vector<T>& v, uint64_t limit = UINT64_MAX) {
        uint64_t n = 0;
        if (!count(n, sizeof(T), limit)) return false;
        v.resize((size_t) n);
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    template<class T> bool fixed_vec(std::vector<T>& v, uint64_t expected) {
        uint64_t n = 0;
        if (!count(n, sizeof(T), expected)) return false;
        if (n != expected) return fail("DFlash2 array size differs from its shape");
        v.resize((size_t) n);
        return raw(v.data(), (size_t) n * sizeof(T));
    }
    bool buffer(ConversationBuffer& b, size_t part) {
        uint64_t n = 0;
        const uint64_t limit = kv_layer < limits.max_kv_bytes.size() ? limits.max_kv_bytes[kv_layer][part] : UINT64_MAX;
        if (!count(n, 1, limit)) return false;
        b = {};
        b.resize((size_t) n);
        return b.visit(0, b.size(), [&](uint8_t* p, size_t c, size_t) { return raw(p, c); });
    }
};

bool get_checkpoint(In& in, ConversationCheckpoint& c) {
    if (!in.vec(c.ids, in.limits.max_tokens)) return false;
    uint64_t n = 0;
    if (!in.count(n, 16, in.limits.max_tokens)) return false;
    c.imgs.resize((size_t) n);
    for (auto& k : c.imgs) if (!in.i64(k.start) || !in.u64(k.hash)) return false;
    const auto& m = in.limits.max_state_bytes;   // byte arrays: element and byte counts are the same
    return in.vec(c.gdn, m[0]) && in.vec(c.ple, m[1]) && in.vec(c.tails, m[2]) && in.vec(c.dead, m[3]) &&
           in.vec(c.block_pos, m[4]) && in.u64(c.used);
}

bool get_dflash2(In& in, Dflash2State& s) {
    uint64_t version = 0, q4 = 0, enabled = 0;
    if (!in.u64(version)) return false;
    if (version != 1) return in.fail("unsupported DFlash2 state version");
    auto& g = s.shape;
    for (int64_t* v : {&g.hidden, &g.layers, &g.heads, &g.kv_heads, &g.head_dim, &g.inter, &g.window, &g.group, &g.rank,
                       &g.mask_token, &g.vocab, &g.ring_slots, &g.deferred_slots, &g.log_chunk_rows})
        if (!in.i64(*v)) return false;
    std::string why;
    if (!dflash2_shape_validate(g, why)) return in.fail(why);
    if (!in.raw(&g.theta, sizeof(float)) || !in.raw(&g.eps, sizeof(float)) || !in.u64(q4) ||
        !in.vec(g.taps, in.limits.dflash2 ? in.limits.dflash2->shape.taps.size() : UINT64_MAX)) return false;
    if (q4 > 1) return in.fail("invalid DFlash2 weight format flag");
    g.q4 = q4 == 1;
    if (in.limits.dflash2 && !(g == in.limits.dflash2->shape)) return in.fail("DFlash2 runtime shape differs");
    if (!in.u64(enabled) || !in.i64(s.epoch) || !in.i64(s.have_ctx) || !in.i64(s.ring_hi) ||
        !in.i64(s.deferred_lo) || !in.i64(s.deferred_hi)) return false;
    if (enabled > 1) return in.fail("invalid DFlash2 ring-log flag");
    s.log_enabled = enabled == 1;
    const uint64_t tokens = std::min<uint64_t>({in.limits.max_tokens,
        in.limits.dflash2 ? in.limits.dflash2->max_tokens : uint64_t(INT32_MAX), uint64_t(INT32_MAX)});
    if (in.limits.dflash2 && s.log_enabled != in.limits.dflash2->log_enabled) return in.fail("DFlash2 ring-log mode differs");
    if (s.epoch < 0 || s.epoch > INT32_MAX || s.have_ctx < 0 || (uint64_t) s.have_ctx > tokens ||
        s.ring_hi < 0 || (uint64_t) s.ring_hi > tokens || s.deferred_lo < 0 || s.deferred_hi < s.deferred_lo ||
        (uint64_t) s.deferred_hi > tokens || s.deferred_hi - s.deferred_lo > g.deferred_slots ||
        (s.deferred_hi == s.deferred_lo && s.deferred_lo != 0)) return in.fail("invalid DFlash2 counters");
    uint64_t deferred_elements = 0;
    if (!dflash2_deferred_elements(g, (uint64_t) (s.deferred_hi - s.deferred_lo), deferred_elements))
        return in.fail("DFlash2 deferred byte count overflow");
    const uint64_t per = (uint64_t) g.ring_slots * (uint64_t) g.kv_heads * (uint64_t) g.head_dim;
    if (!in.fixed_vec(s.ring, (uint64_t) g.layers * 2 * per) || !in.fixed_vec(s.tags, g.ring_slots) ||
        !in.fixed_vec(s.slot_positions, g.ring_slots) || !in.vec(s.history, tokens)) return false;
    const uint64_t chunks = (tokens + (uint64_t) g.log_chunk_rows - 1) / (uint64_t) g.log_chunk_rows;
    const uint64_t nq = (uint64_t) g.log_chunk_rows * (uint64_t) g.hidden, nd = nq / 32;
    uint64_t n = 0;
    if (!in.count(n, (size_t) (32 + nq + nd * sizeof(float) + g.log_chunk_rows), s.log_enabled ? chunks : 0)) return false;
    s.log.resize((size_t) n);
    uint64_t previous = 0;
    for (size_t i = 0; i < s.log.size(); ++i) {
        auto& c = s.log[i];
        if (!in.u64(c.index)) return false;
        if (c.index >= chunks || (i && c.index <= previous)) return in.fail("invalid DFlash2 log chunk index");
        previous = c.index;
        if (!in.fixed_vec(c.q, nq) || !in.fixed_vec(c.scales, nd) || !in.fixed_vec(c.valid, g.log_chunk_rows)) return false;
    }
    if (!in.fixed_vec(s.deferred, deferred_elements)) return false;
    return dflash2_state_validate(s, why, tokens) || in.fail(why);
}

bool get_payload(In& in, SavedConversation& s) {
    for (auto& g : s.geometry) if (!in.i64(g)) return false;
    // checked before any state array is read: a file of another geometry allocates nothing
    if (in.limits.geometry && *in.limits.geometry != s.geometry)
        return in.fail("saved with another model geometry than this runtime's");
    uint64_t cvec = 0;
    if (!in.i64(s.layer_lo) || !in.i64(s.layer_hi) || !in.u64(cvec)) return false;
    if (in.limits.layer_range &&
        (in.limits.layer_range->first != s.layer_lo || in.limits.layer_range->second != s.layer_hi))
        return in.fail("saved with another layer range than this runtime's");
    if (cvec > 1) return in.fail("invalid cvec flag");
    s.cvec = cvec == 1;
    if (!get_checkpoint(in, s.live)) return false;
    uint64_t n = 0;
    // a checkpoint is at least 8 counts + used = 72 bytes; a K/V layer at least 7 + 5 = 96
    if (!in.count(n, 72, in.limits.max_checkpoints)) return false;
    s.checkpoints.resize((size_t) n);
    for (auto& c : s.checkpoints) if (!get_checkpoint(in, c)) return false;
    if (!in.count(n, 96, in.limits.max_kv_layers)) return false;
    s.kv.resize((size_t) n);
    for (auto& k : s.kv) {
        in.kv_layer = (size_t) (&k - s.kv.data());
        int64_t format = 0;
        if (!in.i64(format) || !in.i64(k.cells) || !in.i64(k.heads) || !in.i64(k.head_dim) ||
            !in.i64(k.page_size) || !in.i64(k.pooled_rows) || !in.i64(k.idx_dim)) return false;
        if (format < INT32_MIN || format > INT32_MAX) return in.fail("invalid K/V format");
        k.format = (int) format;
        if (!in.buffer(k.k, 0) || !in.buffer(k.v, 1) || !in.buffer(k.k_scale, 2) || !in.buffer(k.v_scale, 3) ||
            !in.buffer(k.pooled, 4)) return false;
    }
    return true;
}

bool has_stage_parts(const SavedConversation& s) {
    if (!s.live.stage_parts.empty()) return true;
    for (const auto& c : s.checkpoints) if (!c.stage_parts.empty()) return true;
    return false;
}

// header v1, little-endian: magic[8] version:u32 header_size:u32 model:u64 config:u64 payload:u64 reserved:u64[2]
// header_hash:u64 (session_hash64 of bytes 0..55, seed 0)
void header_bytes(uint8_t* h, const SessionFileIdentity& id, uint64_t payload, uint32_t version = kVersion) {
    std::memset(h, 0, kHeader);
    std::memcpy(h, kMagic, 8);
    const uint32_t size = kHeader;
    std::memcpy(h + 8, &version, 4);
    std::memcpy(h + 12, &size, 4);
    std::memcpy(h + 16, &id.model, 8);
    std::memcpy(h + 24, &id.config, 8);
    std::memcpy(h + 32, &payload, 8);
    const uint64_t hh = session_hash64(h, 56, 0);
    std::memcpy(h + 56, &hh, 8);
}
} // namespace

SessionHasher::SessionHasher(uint64_t seed)
    : lane_{seed + P1 + P2, seed + P2, seed, seed - P1}, pending_{} {}

void SessionHasher::block(const uint8_t* p) {
    for (int i = 0; i < 4; ++i) lane_[i] = round1(lane_[i], load64(p + 8 * i));
}

void SessionHasher::update(const void* data, size_t n) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    total_ += n;
    if (npending_) {
        const size_t take = std::min(n, sizeof pending_ - npending_);
        std::memcpy(pending_ + npending_, p, take);
        npending_ += take; p += take; n -= take;
        if (npending_ < sizeof pending_) return;
        block(pending_);
        npending_ = 0;
    }
    for (; n >= 32; p += 32, n -= 32) block(p);
    if (n) std::memcpy(pending_, p, n);
    npending_ = n;
}

uint64_t SessionHasher::digest() const {
    uint64_t h = rotl(lane_[0], 1) + rotl(lane_[1], 7) + rotl(lane_[2], 12) + rotl(lane_[3], 18);
    for (uint64_t l : lane_) { h ^= round1(0, l); h = h * P1 + P4; }
    h += total_;
    size_t i = 0;
    for (; i + 8 <= npending_; i += 8) { h ^= round1(0, load64(pending_ + i)); h = rotl(h, 27) * P1 + P4; }
    for (; i < npending_; ++i) { h ^= pending_[i] * P5; h = rotl(h, 11) * P1; }
    h ^= h >> 33; h *= P2; h ^= h >> 29; h *= P3; h ^= h >> 32;
    return h;
}

uint64_t session_hash64(const void* data, size_t n, uint64_t seed) {
    SessionHasher h(seed);
    h.update(data, n);
    return h.digest();
}

void SessionIdentityBuilder::field(const std::string& tag, uint8_t type, const void* data, uint64_t n) {
    const uint64_t tn = tag.size();
    hash_.update(&tn, 8);
    hash_.update(tag.data(), tag.size());
    hash_.update(&type, 1);
    hash_.update(&n, 8);
    if (n) hash_.update(data, (size_t) n);
}
void SessionIdentityBuilder::str(const std::string& tag, const std::string& v) { field(tag, 's', v.data(), v.size()); }
void SessionIdentityBuilder::i64(const std::string& tag, int64_t v) { field(tag, 'i', &v, 8); }
void SessionIdentityBuilder::u64(const std::string& tag, uint64_t v) { field(tag, 'u', &v, 8); }
void SessionIdentityBuilder::f64(const std::string& tag, double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, 8);
    field(tag, 'f', &bits, 8);
}
void SessionIdentityBuilder::bytes(const std::string& tag, const void* data, size_t n) { field(tag, 'b', data, n); }

uint64_t session_config_fingerprint(const SessionConfig& c) {
    SessionIdentityBuilder b(0x434f4e4649473031ull);   // "CONFIG01"
    b.str("engine", c.engine_version);
    b.str("backend", c.backend);
    b.str("kv", c.kv);
    b.i64("max_context", c.max_context);
    b.i64("kv_resident", c.kv_resident);
    b.i64("mtp_window", c.mtp_window);
    b.i64("kv_rot", c.kv_rot ? 1 : 0);
    b.i64("rope.type", c.rope.type);
    b.f64("rope.freq_base", c.rope.freq_base);
    b.f64("rope.factor", c.rope.factor);
    b.f64("rope.freq_scale", c.rope.freq_scale);
    b.f64("rope.orig_ctx", c.rope.orig_ctx);
    b.f64("rope.ext_factor", c.rope.ext_factor);
    b.f64("rope.attn_factor", c.rope.attn_factor);
    b.f64("rope.beta_fast", c.rope.beta_fast);
    b.f64("rope.beta_slow", c.rope.beta_slow);
    b.u64("cvec", c.cvec);
    b.u64("switches", c.switches.size());
    for (const auto& [name, value] : c.switches) b.i64("switch." + name, value);
    return b.digest();
}

const char* session_error_name(SessionError e) {
    switch (e) {
        case SessionError::none: return "none";
        case SessionError::invalid: return "invalid";
        case SessionError::storage: return "storage";
        case SessionError::memory: return "memory";
        case SessionError::io: return "io";
    }
    return "io";
}

std::string session_free_space_dir(const std::string& path, bool windows_rules) {
    if (!windows_rules) {
        const size_t cut = path.find_last_of('/');
        if (cut == std::string::npos) return ".";
        return path.substr(0, cut == 0 ? 1 : cut);
    }
    std::string p = path;
    for (char& c : p) if (c == '/') c = '\\';
    const size_t cut = p.find_last_of('\\');
    if (cut == std::string::npos) {
        // "C:chat.bin" (drive-relative): the drive root answers for the same volume
        if (p.size() >= 2 && p[1] == ':') return p.substr(0, 2) + "\\";
        return std::string();   // the current folder
    }
    return p.substr(0, cut + 1);   // with its trailing backslash: "\\server\share\dir\", "C:\"
}

std::vector<SessionModelFile> session_model_inputs(const SessionInputs& in) {
    std::vector<SessionModelFile> f;
    for (size_t i = 0; i < in.native_shards.size(); ++i) f.push_back({"native shard " + std::to_string(i), in.native_shards[i]});
    for (size_t i = 0; i < in.native_dense.size(); ++i) f.push_back({"native dense " + std::to_string(i), in.native_dense[i]});
    for (size_t i = 0; i < in.native_head.size(); ++i) f.push_back({"native head " + std::to_string(i), in.native_head[i]});
    if (!in.embedding.empty()) f.push_back({"embedding", in.embedding});
    if (!in.ple.empty()) f.push_back({"ple", in.ple});
    if (!in.mtp.empty()) f.push_back({"mtp", in.mtp});
    if (!in.pack.empty()) f.push_back({"pack native_experts.txt", in.pack + "/native_experts.txt", true});
    for (const auto& [role, path] : in.experts) f.push_back({role, path});
    return f;
}

bool session_model_fingerprint(const std::vector<SessionModelFile>& files, uint64_t& fingerprint, std::string& error,
                               const std::function<void()>& beat) {
    SessionIdentityBuilder h(0x5354524154414d44ull);
    std::vector<uint8_t> buf(1u << 20);
    // a file in several roles is read once; its sample enters the fingerprint under every role
    std::unordered_map<std::string, std::pair<bool, uint64_t>> seen;   // path -> (present, sample digest)
    h.u64("files", files.size());
    for (const auto& file : files) {
        auto it = seen.find(file.path);
        if (it == seen.end()) {
            RawFile f;
            uint64_t size = 0;
            std::pair<bool, uint64_t> d{false, 0};
            if (!f.open_model(file.path, size)) {
                if (!(file.optional && f.missing())) {
                    error = "session fingerprint: " + file.role + " " + file.path + ": " + f.error();
                    return false;
                }
            } else {
                SessionIdentityBuilder one(0x46494c4553414d50ull);   // "FILESAMP"
                one.u64("size", size);
                const uint64_t head = std::min<uint64_t>(size, buf.size());
                if (!f.read_at(0, buf.data(), (size_t) head)) {
                    error = "session fingerprint: reading " + file.path + ": " + f.error();
                    return false;
                }
                one.bytes("head", buf.data(), (size_t) head);
                if (size > buf.size()) {
                    const uint64_t tail = std::min<uint64_t>(size - head, buf.size());
                    if (!f.read_at(size - tail, buf.data(), (size_t) tail)) {
                        error = "session fingerprint: reading " + file.path + ": " + f.error();
                        return false;
                    }
                    one.bytes("tail", buf.data(), (size_t) tail);
                }
                d = {true, one.digest()};
            }
            it = seen.emplace(file.path, d).first;
            if (beat) beat();
        }
        if (!it->second.first) {
            if (!file.optional) { error = "session fingerprint: " + file.role + " " + file.path + " is missing"; return false; }
            h.str("absent", file.role);
            continue;
        }
        h.str("role", file.role);
        h.u64("sample", it->second.second);
    }
    fingerprint = h.digest();
    return true;
}

namespace {
uint64_t sat_add(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sat_mul(uint64_t a, uint64_t b) { return a && b > UINT64_MAX / a ? UINT64_MAX : a * b; }
} // namespace

uint64_t session_read_max_file_bytes(const SessionReadLimits& l) {
    const uint64_t open = UINT64_MAX;
    if (l.max_tokens == open || l.max_checkpoints == open || l.max_kv_layers == open) return l.max_file_bytes;
    for (uint64_t b : l.max_state_bytes) if (b == open) return l.max_file_bytes;
    if (l.max_kv_bytes.size() < l.max_kv_layers) return l.max_file_bytes;
    // one checkpoint: ids (8 + 4 per token), images (8 + 16 per image), five state arrays (8 + bytes), used (8)
    uint64_t cp = sat_add(16, sat_mul(l.max_tokens, 4 + 16));
    for (uint64_t b : l.max_state_bytes) cp = sat_add(cp, sat_add(8, b));
    cp = sat_add(cp, 8);
    uint64_t n = kHeader + kTrailer + 18 * 8 + 3 * 8;             // geometry, layer range, cvec
    n = sat_add(n, cp);                                            // live
    n = sat_add(n, sat_add(8, sat_mul(l.max_checkpoints, cp)));    // checkpoints
    n = sat_add(n, 8);
    for (uint64_t i = 0; i < l.max_kv_layers; ++i) {
        n = sat_add(n, 7 * 8);
        for (uint64_t b : l.max_kv_bytes[(size_t) i]) n = sat_add(n, sat_add(8, b));
    }
    if (l.dflash2) {
        const auto& d = *l.dflash2;
        std::string why;
        if (!dflash2_shape_validate(d.shape, why)) return l.max_file_bytes;
        const auto& g = d.shape;
        const uint64_t tokens = std::min<uint64_t>({l.max_tokens, d.max_tokens, INT32_MAX});
        const uint64_t chunks = (tokens + (uint64_t) g.log_chunk_rows - 1) / (uint64_t) g.log_chunk_rows;
        uint64_t extension = sat_add(240, sat_mul(g.taps.size(), sizeof(int32_t)));
        extension = sat_add(extension, sat_mul((uint64_t) g.layers * 2 * g.kv_heads * g.ring_slots * g.head_dim, sizeof(float)));
        extension = sat_add(extension, (uint64_t) g.ring_slots * 2 * sizeof(int64_t));
        extension = sat_add(extension, sat_mul(tokens, sizeof(int32_t)));
        if (d.log_enabled) extension = sat_add(extension, sat_mul(chunks,
            32 + (uint64_t) g.log_chunk_rows * ((uint64_t) g.hidden + (uint64_t) g.hidden / 32 * sizeof(float) + 1)));
        extension = sat_add(extension, sat_mul(sat_mul(sat_mul(std::min<uint64_t>(tokens, g.deferred_slots),
                                       g.taps.size()), (uint64_t) g.hidden), sizeof(float)));
        n = sat_add(n, extension);
    }
    return std::min(n, l.max_file_bytes);
}

uint64_t session_read_peak_bytes(uint64_t file_bytes) {
    const uint64_t segments = file_bytes / kBlock + 1;
    // the parsed image (about the payload), the read buffer, a page per buffer segment, the small vectors
    return sat_add(sat_add(sat_add(file_bytes, kBlock), sat_mul(segments, 4096)), uint64_t(1) << 20);
}

namespace {
bool write_impl(const std::string& path, const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                const SessionFileIdentity& id, size_t& bytes, std::string& error, const SessionWriteOptions& opt,
                SessionStatus& st) {
    st = {};
    st.error = SessionError::invalid;   // refusals before any disk access
    if (has_stage_parts(image)) { error = "session file: layer-split state cannot be saved"; return false; }
    if (sources && !image.kv.empty()) { error = "session file: K/V given both in the image and as sources"; return false; }
    if (image.dflash2 && !dflash2_state_validate(*image.dflash2, error)) return false;
    Out sizing;
    put_payload(sizing, image, sources);
    if (sizing.overflow || sizing.n > UINT64_MAX - kHeader - kTrailer ||
        sizing.n + kHeader + kTrailer > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "session file: the state is too large for this build";
        return false;
    }
    const uint64_t payload = sizing.n, total = kHeader + payload + kTrailer;
    st.error = SessionError::io;
    if (opt.min_free_bytes) {
        uint64_t avail = 0;
        if (!free_space(path, avail)) { error = "session file: cannot read the free disk space for " + path; return false; }
        if (avail < total || avail - total < opt.min_free_bytes) {
            st.error = SessionError::storage;
            error = "session file: not enough disk space (" + std::to_string(total >> 20) + " MiB plus a reserve of " +
                    std::to_string(opt.min_free_bytes >> 20) + " MiB needed, " + std::to_string(avail >> 20) +
                    " MiB free)";
            return false;
        }
    }
    Publisher pub;
    if (!pub.open(path, error)) { st.error = pub.kind(); return false; }
    RawFile raw;
    if (!pub.create_temp(raw, error)) {
        st.error = raw.kind();
        error = "session file: cannot create a temporary file beside " + path + ": " + error;
        return false;
    }
    FileSink f(raw);
    if (!f.ok()) { st.error = SessionError::memory; error = "session file: " + f.error(); return false; }
    if (opt.progress) f.progress.emplace(opt.progress);
    f.phase = opt.phase;
    f.expected = total;
    if (f.progress) (*f.progress)(0, total);   // the temporary file exists: the transfer starts
    const std::function<int(const char*)>* fault = opt.fault ? &opt.fault : nullptr;
    f.write_fault = injected(fault, "write");
    f.flush_fault = injected(fault, "file_flush");
    uint8_t h[kHeader];
    header_bytes(h, id, payload, image.dflash2 ? 2 : kVersion);
    bool ok = f.write(h, kHeader);
    SessionHasher hash(0);
    Out out;
    out.f = &f;
    out.hash = &hash;
    if (ok) put_payload(out, image, sources);
    ok = ok && out.ok && out.n == payload;
    if (ok) {
        const uint64_t ph = hash.digest();
        ok = f.write(&ph, 8) && f.write(kEnd, 8);
    }
    ok = f.finish(opt.durable) && ok;
    std::string why;
    if (!ok) {
        why = out.read_failed ? "K/V source read" + (out.read_error.empty() ? std::string() : ": " + out.read_error)
                              : f.error();
        st.error = out.read_failed ? SessionError::io : f.kind();
    } else {
        if (opt.phase) opt.phase("publish", total);
        if (pub.publish(why, opt.durable, st, fault)) {
            st.error = SessionError::none;
            bytes = (size_t) total;
            return true;
        }
    }
    error = "session file: writing " + path + " failed" + (why.empty() ? std::string() : ": " + why);
    if (st.published) error += " (the new file has already replaced the old one; its folder entry may not survive a power loss)";
    return false;   // Publisher removes this call's temporary file, and only that
}

bool write_guarded(const std::string& path, const SavedConversation& image, const std::vector<SessionKvSource>* sources,
                   const SessionFileIdentity& id, size_t& bytes, std::string& error, const SessionWriteOptions& opt,
                   SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = write_impl(path, image, sources, id, bytes, error, opt, st);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while writing " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}
} // namespace

std::vector<ConversationCheckpoint> session_checkpoints_to_save(const std::vector<ConversationCheckpoint>& chain) {
    const ConversationCheckpoint* deepest = session_deepest_checkpoint(chain);
    if (!deepest) return {};
    return {*deepest};
}

const ConversationCheckpoint* session_deepest_checkpoint(const std::vector<ConversationCheckpoint>& chain) {
    // A pinned shared prefix (pin=N) is the resume point the next queries want, so with one in the chain the file
    // keeps the deepest PINNED checkpoint, not a deeper periodic one a long private suffix made past it.  No pin:
    // the deepest, as before.
    const ConversationCheckpoint* deepest = nullptr;
    bool any_pinned = false;
    for (const auto& c : chain) any_pinned = any_pinned || c.pinned;
    for (const auto& c : chain)
        if ((!any_pinned || c.pinned) && (!deepest || c.ids.size() > deepest->ids.size())) deepest = &c;
    return deepest;
}

uint64_t session_save_peak_bytes(const ConversationCheckpoint* deepest, const SessionSaveLive& live) {
    uint64_t n = sat_add(kBlock, uint64_t(1) << 20);                          // write buffer, small vectors
    if (deepest) n = sat_add(n, sat_mul(2, deepest->bytes()));                // the list and the image's copy
    n = sat_add(n, live.state_bytes);                                         // the live running state
    n = sat_add(n, live.dflash2_bytes);
    n = sat_add(n, sat_mul(live.tokens, sizeof(int32_t)));                    // its token list
    n = sat_add(n, sat_mul(live.images, sizeof(ConversationImageKey)));       // its images
    n = sat_add(n, sat_mul(live.kv_layers, sizeof(SessionKvSource) + 256));   // the K/V source directory
    return n;
}

bool session_save_checkpoints(const std::vector<ConversationCheckpoint>& chain, const SessionSaveLive& live,
                              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit,
                              std::vector<ConversationCheckpoint>& out, std::string& why) {
    out.clear();
    const ConversationCheckpoint* deepest = session_deepest_checkpoint(chain);
    if (admit && !admit(session_save_peak_bytes(deepest, live), why)) return false;
    if (deepest) out.push_back(*deepest);
    return true;
}

int64_t session_phase_limit_s(uint64_t bytes) {
    return std::min<int64_t>(3600, 60 + (int64_t) std::min<uint64_t>(bytes >> 22, 3600));
}

bool session_file_write(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                        const SessionFileIdentity& id, size_t& bytes, std::string& error,
                        const SessionWriteOptions& options, SessionStatus* status) {
    return write_guarded(path, meta, &kv, id, bytes, error, options, status);
}

bool session_file_write(const std::string& path, const SavedConversation& image, const SessionFileIdentity& id,
                        size_t& bytes, std::string& error, const SessionWriteOptions& options, SessionStatus* status) {
    return write_guarded(path, image, nullptr, id, bytes, error, options, status);
}

namespace {
bool read_impl(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
               size_t& bytes, std::string& error, const SessionReadLimits& limits, SessionStatus& st) {
    error.clear();
    st = {};
    st.error = SessionError::invalid;   // every refusal below, unless the step says otherwise
    RawFile raw;
    uint64_t size = 0;
    if (!raw.open_session(path, size)) {
        st.error = raw.refused() ? SessionError::invalid : raw.kind();
        error = "session file: " + path + ": " + raw.error();
        return false;
    }
    if (size < kHeader + kTrailer) { error = "session file: size " + std::to_string(size) + " is below the minimum"; return false; }
    const uint64_t max_size = limits.max_file_bytes;
    if (size > max_size || size > (uint64_t) std::numeric_limits<size_t>::max()) {
        error = "session file: size " + std::to_string(size) + " exceeds this engine's limit" +
                (max_size == UINT64_MAX ? std::string() : " of " + std::to_string(max_size));
        return false;
    }
    FileSource f(raw);
    if (!f.ok()) { st.error = SessionError::memory; error = "session file: " + f.error(); return false; }
    // installed before the first block (the header's) is read: every block reports, the first one included
    if (limits.progress) f.progress.emplace(limits.progress);
    f.expected = size;
    uint8_t h[kHeader];
    if (!f.read(h, kHeader)) { st.error = f.kind(); error = "session file: header read error: " + f.error(); return false; }
    uint32_t version = 0, hsize = 0;
    uint64_t model = 0, config = 0, payload = 0, r0 = 0, r1 = 0, hh = 0;
    std::memcpy(&version, h + 8, 4); std::memcpy(&hsize, h + 12, 4);
    std::memcpy(&model, h + 16, 8); std::memcpy(&config, h + 24, 8); std::memcpy(&payload, h + 32, 8);
    std::memcpy(&r0, h + 40, 8); std::memcpy(&r1, h + 48, 8); std::memcpy(&hh, h + 56, 8);
    if (std::memcmp(h, kMagic, 8) != 0) { error = "session file: not a Strata session file (magic)"; return false; }
    if (hh != session_hash64(h, 56, 0)) { error = "session file: header checksum mismatch"; return false; }
    if (version != kVersion && version != 2) { error = "session file: unsupported version " + std::to_string(version); return false; }
    if (hsize != kHeader || r0 || r1) { error = "session file: invalid header fields"; return false; }
    if (model != id.model) { error = "session file: saved with another model (model fingerprint differs)"; return false; }
    if (config != id.config) { error = "session file: saved with another engine configuration (config fingerprint differs)"; return false; }
    if (payload > size || size - kHeader - kTrailer != payload) {
        error = "session file: size " + std::to_string(size) + " does not match the header (" +
                std::to_string(kHeader + payload + kTrailer) + ")";
        return false;
    }
    // With no DFlash runtime limits, a v2 extension is bounded by its payload and the fixed implementation bounds.
    const uint64_t bounded = version == 2 && !limits.dflash2 ? limits.max_file_bytes : session_read_max_file_bytes(limits);
    if (size > bounded) { error = "session file: size exceeds this engine's limit of " + std::to_string(bounded); return false; }
    if (limits.admit) {
        std::string why;
        if (!limits.admit(session_read_peak_bytes(size), why)) {
            st.error = SessionError::memory;
            error = "session file: " + why;
            return false;
        }
    }
    SavedConversation parsed;
    In in{&f, SessionHasher(0), payload, error, limits};
    if (!get_payload(in, parsed)) { st.error = in.kind; return false; }
    if (version == 2) {
        parsed.dflash2.emplace();
        if (!get_dflash2(in, *parsed.dflash2)) { st.error = in.kind; return false; }
    }
    if (in.left) { error = "session file: payload has " + std::to_string(in.left) + " unparsed bytes"; return false; }
    uint8_t t[kTrailer];
    if (!f.read(t, kTrailer)) { st.error = f.kind(); error = "session file: trailer read error: " + f.error(); return false; }
    uint64_t ph = 0;
    std::memcpy(&ph, t, 8);
    if (ph != in.hash.digest()) { error = "session file: payload checksum mismatch"; return false; }
    if (std::memcmp(t + 8, kEnd, 8) != 0) { error = "session file: bad end marker"; return false; }
    image = std::move(parsed);
    bytes = (size_t) size;
    st.error = SessionError::none;
    return true;
}
} // namespace

bool session_file_read(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                       size_t& bytes, std::string& error, const SessionReadLimits& limits, SessionStatus* status) {
    SessionStatus st;
    bool ok = false;
    try {
        ok = read_impl(path, id, image, bytes, error, limits, st);
    } catch (const std::bad_alloc&) {
        error = "session file: out of memory while reading " + path;
        st.error = SessionError::memory;
        ok = false;
    }
    if (status) *status = st;
    return ok;
}

namespace {
namespace fs = std::filesystem;
fs::path block_fs_path(const std::string& path) {
    std::u8string utf8(path.size(), u8'\0');
    std::memcpy(utf8.data(), path.data(), path.size());
    return fs::path(utf8);
}
std::string block_fs_string(const fs::path& path) {
    const auto utf8 = path.u8string();
    return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
}
constexpr char kManifestMagic[8] = {'S','T','R','M','A','N','F','\x02'};
constexpr char kChunkMagic[8] = {'S','T','R','K','V','B','L','\x01'};
constexpr uint64_t kContentSeed = 0x535452424c4f434bull;
constexpr uint64_t kChunkCells = 4096;

void block_header(uint8_t* header, const char* magic, const SessionFileIdentity& id, uint64_t payload,
                  uint64_t metadata_bytes = 0, uint64_t metadata_hash = 0, uint32_t version = 2) {
    header_bytes(header, id, payload);
    std::memcpy(header, magic, 8);
    if (metadata_bytes) {
        std::memcpy(header + 8, &version, 4);
        std::memcpy(header + 40, &metadata_bytes, 8);
        std::memcpy(header + 48, &metadata_hash, 8);
    }
    const uint64_t hh = session_hash64(header, 56, 0);
    std::memcpy(header + 56, &hh, 8);
}

std::string block_hex(uint64_t n) {
    char s[17];
    std::snprintf(s, sizeof s, "%016llx", (unsigned long long) n);
    return s;
}
struct BlockRef {
    uint64_t hash = 0, hash2 = 0, bytes = 0;
    std::string name() const { return block_hex(hash) + "-" + block_hex(hash2) + "-" + block_hex(bytes) + ".block"; }
};
std::string manifest_name(const std::string& key) {
    return block_hex(session_hash64(key.data(), key.size(), 0)) + "-" +
           block_hex(session_hash64(key.data(), key.size(), kContentSeed)) + ".manifest";
}
bool named_hex_file(const std::string& name, size_t fields, const char* suffix) {
    const size_t body = fields * 16 + fields - 1;
    if (name.size() != body + std::strlen(suffix) || name.compare(body, std::string::npos, suffix)) return false;
    for (size_t i = 0; i < body; ++i) {
        if (i % 17 == 16) { if (name[i] != '-') return false; }
        else if (!((name[i] >= '0' && name[i] <= '9') || (name[i] >= 'a' && name[i] <= 'f'))) return false;
    }
    return true;
}
bool block_fail(SessionStatus& st, std::string& error, const std::string& why,
                SessionError kind = SessionError::invalid) {
    st.error = kind; error = "session blocks: " + why; return false;
}
bool block_dirs(const std::string& directory, bool create, std::string& error, SessionStatus& st) {
    const fs::path root = block_fs_path(directory).parent_path();
    if (create) fs::create_directories(root);
    for (const auto& path : {block_fs_path(directory), block_fs_path(directory) / "manifests",
                             block_fs_path(directory) / "blocks"}) {
        if (create) fs::create_directory(path);
        const auto type = fs::symlink_status(path).type();
        if (type == fs::file_type::not_found && !create) return true;
        if (type != fs::file_type::directory) return block_fail(st, error, "managed directory is not a directory");
    }
    if (create) {
        // A previous attempt may have created these names but failed their flush. Repair every required parent
        // barrier on retry too; existence in this process does not prove a name was durably published.
        std::vector<fs::path> names{block_fs_path(directory) / "manifests"};
        for (fs::path path = block_fs_path(directory); !path.empty() && path != path.parent_path();
             path = path.parent_path()) names.push_back(path);
        for (const auto& path : names) {
            Publisher parent;
            if (!parent.open(block_fs_string(path), error)) { st.error = parent.kind(); return false; }
            if (!parent.sync_directory(error, st)) return false;
        }
    }
    return true;
}

// Separate format magic, same bounded stream/checksum/atomic publication and fault semantics as manual sessions.
bool block_write_file(const std::string& path, const char* magic, const SessionFileIdentity& id,
                      const std::function<void(Out&)>& put, uint64_t& bytes, std::string& error,
                      const SessionWriteOptions& opt, SessionStatus& st,
                      uint64_t metadata_bytes = 0, uint64_t metadata_hash = 0, uint32_t version = 2) {
    Out sizing; put(sizing);
    if (!sizing.ok || sizing.n > SIZE_MAX - kHeader - kTrailer)
        return block_fail(st, error, "payload too large");
    const uint64_t total = sizing.n + kHeader + kTrailer;
    if (opt.min_free_bytes) {
        uint64_t available = 0;
        if (!free_space(path, available)) return block_fail(st, error, "cannot read free disk space", SessionError::io);
        if (available < total || available - total < opt.min_free_bytes)
            return block_fail(st, error, "not enough disk space", SessionError::storage);
    }
    Publisher pub;
    if (!pub.open(path, error)) { st.error = pub.kind(); return false; }
    RawFile raw;
    if (!pub.create_temp(raw, error)) { st.error = raw.kind(); return false; }
    FileSink sink(raw);
    if (!sink.ok()) return block_fail(st, error, sink.error(), SessionError::memory);
    if (opt.progress) sink.progress.emplace(opt.progress);
    sink.expected = total; sink.phase = opt.phase;
    if (sink.progress) (*sink.progress)(0, total);
    const auto* fault = opt.fault ? &opt.fault : nullptr;
    sink.write_fault = injected(fault, "write"); sink.flush_fault = injected(fault, "file_flush");
    uint8_t header[kHeader];
    block_header(header, magic, id, sizing.n, metadata_bytes, metadata_hash, version);
    SessionHasher hash;
    Out out; out.f = &sink; out.hash = &hash;
    bool ok = sink.write(header, sizeof header);
    if (ok) put(out);
    ok = ok && out.ok && out.n == sizing.n;
    const uint64_t checksum = hash.digest();
    ok = ok && sink.write(&checksum, 8) && sink.write(kEnd, 8);
    ok = sink.finish(opt.durable) && ok;
    if (!ok) return block_fail(st, error, "write failed: " + sink.error(), sink.kind());
    if (opt.phase) opt.phase("publish", total);
    if (!pub.publish(error, opt.durable, st, fault)) return false;
    st.error = SessionError::none; bytes = total; return true;
}

bool block_read_file(const std::string& path, const char* magic, const SessionFileIdentity& id,
                     const SessionReadLimits& limits, uint64_t max_bytes,
                     const std::function<bool(In&)>& get, uint64_t& bytes, std::string& error, SessionStatus& st) {
    RawFile raw;
    uint64_t size = 0;
    if (!raw.open_session(path, size))
        return block_fail(st, error, raw.error(), raw.refused() ? SessionError::invalid : raw.kind());
    if (size < kHeader + kTrailer || size > max_bytes || size > SIZE_MAX)
        return block_fail(st, error, "file size exceeds bounds");
    FileSource source(raw);
    if (!source.ok()) return block_fail(st, error, source.error(), SessionError::memory);
    if (limits.progress) source.progress.emplace(limits.progress);
    source.expected = size;
    uint8_t h[kHeader], expected[kHeader];
    if (!source.read(h, sizeof h)) return block_fail(st, error, source.error(), source.kind());
    const uint64_t payload = load64(h + 32);
    block_header(expected, magic, id, payload);
    if (std::memcmp(h, expected, kHeader) || payload != size - kHeader - kTrailer)
        return block_fail(st, error, "invalid header, identity or size");
    In in{&source, SessionHasher(0), payload, error, limits};
    if (!get(in)) { st.error = in.kind; return false; }
    if (in.left) return block_fail(st, error, "unparsed payload bytes");
    uint8_t trailer[kTrailer];
    if (!source.read(trailer, sizeof trailer)) return block_fail(st, error, source.error(), source.kind());
    if (load64(trailer) != in.hash.digest() || std::memcmp(trailer + 8, kEnd, 8))
        return block_fail(st, error, "payload checksum or trailer mismatch");
    bytes = size; st.error = SessionError::none; return true;
}
std::array<const std::vector<uint8_t>*, 5> block_state_parts(const ConversationCheckpoint& c) {
    return {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
}
std::array<std::vector<uint8_t>*, 5> block_state_parts(ConversationCheckpoint& c) {
    return {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
}
void block_put_checkpoint_metadata(Out& out, const ConversationCheckpoint& c) {
    out.vec(c.ids);
    out.u64(c.imgs.size());
    for (const auto& image : c.imgs) { out.i64(image.start); out.u64(image.hash); }
    for (const auto* part : block_state_parts(c)) out.u64(part->size());
    out.u64(c.used);
}
bool block_checkpoint(In& in, ConversationCheckpoint* full, SessionBlockCheckpoint& entry,
                      std::array<uint64_t, 5>& sizes, uint64_t& private_bytes, uint64_t budget) {
    auto& ids = full ? full->ids : entry.ids;
    auto& images = full ? full->imgs : entry.imgs;
    if (!in.vec(ids, in.limits.max_tokens)) return false;
    uint64_t n = 0;
    if (!in.count(n, 16, in.limits.max_tokens)) return false;
    images.resize((size_t) n);
    for (auto& image : images) if (!in.i64(image.start) || !in.u64(image.hash)) return false;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (!in.u64(n)) return false;
        if (n > in.limits.max_state_bytes[i] || n > SIZE_MAX || n > budget - private_bytes)
            return in.fail("private state exceeds this engine's byte limit");
        sizes[i] = n; private_bytes += n;
    }
    return in.u64(full ? full->used : entry.used);
}
std::array<const ConversationBuffer*, 5> block_parts(const ConversationKv& kv) {
    return {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
}
std::array<ConversationBuffer*, 5> block_parts(ConversationKv& kv) {
    return {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
}
struct BlockPart {
    uint64_t total = 0, chunk = 0;
    std::vector<BlockRef> refs;
};
struct BlockLayer { std::array<BlockPart, 5> parts; };
struct BlockManifest {
    SessionBlockEntry entry;
    uint64_t original_bytes = 0, private_bytes = 0, dflash2_bytes = 0;
    std::vector<std::array<uint64_t, 5>> state_sizes; // live, then checkpoint order
    std::vector<BlockLayer> layers;
};
uint64_t block_chunk_bytes(size_t layer, size_t part, uint64_t total, int64_t cells, SessionBlockLayout layout) {
    if (layer >= layout.main_kv_layers || part >= 4 || !total || cells <= 0) return 0;
    if (total % (uint64_t) cells) return UINT64_MAX;
    return sat_mul(total / (uint64_t) cells, kChunkCells);
}
uint64_t block_whole_chunks(int64_t cells, size_t tokens) {
    return cells <= 0 ? 0 : std::min<uint64_t>((uint64_t) cells, tokens) / kChunkCells;
}
void block_put_metadata(Out& out, const std::string& key, const SavedConversation& image, const BlockManifest& plan) {
    out.u64(plan.original_bytes);
    out.u64(key.size()); out.raw(key.data(), key.size());
    out.u64(plan.entry.layout.main_kv_layers);
    for (int64_t g : image.geometry) out.i64(g);
    out.i64(image.layer_lo); out.i64(image.layer_hi); out.u64(image.cvec ? 1 : 0);
    block_put_checkpoint_metadata(out, image.live);
    out.u64(image.checkpoints.size());
    for (const auto& checkpoint : image.checkpoints) block_put_checkpoint_metadata(out, checkpoint);
    out.u64(image.kv.size());
    for (size_t layer = 0; layer < image.kv.size(); ++layer) {
        const auto& kv = image.kv[layer];
        put_kv_header(out, kv.format, kv.cells, kv.heads, kv.head_dim, kv.page_size, kv.pooled_rows, kv.idx_dim);
        for (size_t part = 0; part < 5; ++part) {
            const auto& p = plan.layers[layer].parts[part];
            out.u64(p.total); out.u64(p.chunk); out.u64(p.refs.size());
            for (const auto& ref : p.refs) { out.u64(ref.hash); out.u64(ref.hash2); out.u64(ref.bytes); }
            out.u64(p.total - p.chunk * p.refs.size());
        }
    }
    if (plan.dflash2_bytes) out.u64(plan.dflash2_bytes);
}

void block_put_private(Out& out, const SavedConversation& image, const BlockManifest& plan) {
    const auto checkpoint = [&](const ConversationCheckpoint& c) {
        for (const auto* part : block_state_parts(c)) out.raw(part->data(), part->size());
    };
    checkpoint(image.live);
    for (const auto& c : image.checkpoints) checkpoint(c);
    for (size_t layer = 0; layer < image.kv.size(); ++layer) {
        const auto parts = block_parts(image.kv[layer]);
        for (size_t part = 0; part < 5; ++part) {
            const auto& p = plan.layers[layer].parts[part];
            const size_t shared = (size_t) (p.chunk * p.refs.size());
            parts[part]->visit(shared, (size_t) p.total - shared,
                              [&](const uint8_t* data, size_t n, size_t) { out.raw(data, n); return out.ok; });
        }
    }
    if (image.dflash2) put_dflash2(out, *image.dflash2);
}

bool block_get_manifest(In& in, const std::string* key, BlockManifest& manifest, SavedConversation* image,
                        bool with_dflash2) {
    uint64_t n = 0, main_layers = 0;
    if (!in.u64(manifest.original_bytes)) return false;
    if (manifest.original_bytes < kHeader + kTrailer ||
        manifest.original_bytes > session_read_max_file_bytes(in.limits))
        return in.fail("logical session exceeds this engine's file limit");
    if (image && in.limits.admit) {
        std::string why;
        if (!in.limits.admit(session_read_peak_bytes(std::max(manifest.original_bytes, in.left)), why)) {
            in.kind = SessionError::memory; return in.fail(why);
        }
    }
    if (!in.count(n, 1)) return false;
    manifest.entry.key.resize((size_t) n);
    if (!in.raw(manifest.entry.key.data(), (size_t) n)) return false;
    if (manifest.entry.key.empty() || (key && manifest.entry.key != *key)) return in.fail("manifest key collision/mismatch");
    if (!in.u64(main_layers) || main_layers > SIZE_MAX) return in.fail("main layer count too large");
    const uint64_t metadata_start = in.left;
    manifest.entry.layout.main_kv_layers = (size_t) main_layers;
    std::array<int64_t, 18> geometry;
    for (auto& g : geometry) if (!in.i64(g)) return false;
    if (in.limits.geometry && *in.limits.geometry != geometry) return in.fail("model geometry mismatch");
    int64_t lo = 0, hi = 0;
    uint64_t cvec = 0;
    if (!in.i64(lo) || !in.i64(hi) || !in.u64(cvec) || cvec > 1) return in.fail("invalid layer range/cvec");
    if (in.limits.layer_range && *in.limits.layer_range != std::pair<int64_t, int64_t>{lo, hi})
        return in.fail("layer range mismatch");
    manifest.entry.cvec = cvec == 1;
    if (image) { image->geometry = geometry; image->layer_lo = lo; image->layer_hi = hi; image->cvec = cvec == 1; }
    manifest.state_sizes.resize(1);
    if (!block_checkpoint(in, image ? &image->live : nullptr, manifest.entry.live, manifest.state_sizes[0],
                          manifest.private_bytes, manifest.original_bytes)) return false;
    if (!in.count(n, 72, in.limits.max_checkpoints)) return false;
    manifest.entry.checkpoints.resize((size_t) n);
    manifest.state_sizes.resize((size_t) n + 1);
    if (image) image->checkpoints.resize((size_t) n);
    for (size_t i = 0; i < n; ++i)
        if (!block_checkpoint(in, image ? &image->checkpoints[i] : nullptr, manifest.entry.checkpoints[i],
                              manifest.state_sizes[i + 1], manifest.private_bytes, manifest.original_bytes)) return false;
    if (!in.count(n, 7 * 8 + 5 * 32, in.limits.max_kv_layers) || main_layers > n) return in.fail("invalid main layer count");
    uint64_t reconstructed = sat_add(kHeader + kTrailer + metadata_start - in.left, manifest.private_bytes);
    if (reconstructed > manifest.original_bytes) return in.fail("metadata exceeds logical session length");
    manifest.entry.kv_layers = (size_t) n;
    manifest.layers.resize((size_t) n);
    if (image) image->kv.resize((size_t) n);
    for (size_t layer = 0; layer < n; ++layer) {
        if (manifest.original_bytes - reconstructed < 96) return in.fail("KV headers exceed logical session length");
        reconstructed += 96;
        ConversationKv header;
        int64_t format = 0;
        if (!in.i64(format) || !in.i64(header.cells) || !in.i64(header.heads) || !in.i64(header.head_dim) ||
            !in.i64(header.page_size) || !in.i64(header.pooled_rows) || !in.i64(header.idx_dim)) return false;
        if (format < INT_MIN || format > INT_MAX || header.cells < 0) return in.fail("invalid KV header");
        header.format = (int) format;
        if (image) image->kv[layer] = std::move(header);
        const int64_t cells = image ? image->kv[layer].cells : header.cells;
        for (size_t part = 0; part < 5; ++part) {
            auto& p = manifest.layers[layer].parts[part];
            uint64_t count = 0, tail = 0;
            if (!in.u64(p.total) || !in.u64(p.chunk) || !in.count(count, 24)) return false;
            const uint64_t limit = layer < in.limits.max_kv_bytes.size() ? in.limits.max_kv_bytes[layer][part] : UINT64_MAX;
            if (p.total > limit || p.total > SIZE_MAX || p.total > manifest.original_bytes - reconstructed)
                return in.fail("KV part exceeds this engine's byte limit");
            reconstructed += p.total;
            const uint64_t chunk = block_chunk_bytes(layer, part, p.total, cells, manifest.entry.layout);
            const uint64_t expected = chunk ? block_whole_chunks(cells, image ? image->live.ids.size() :
                                                                 manifest.entry.live.ids.size()) : 0;
            if (chunk == UINT64_MAX || p.chunk != chunk || count != expected ||
                (count && (p.chunk > p.total / count))) return in.fail("invalid chunk extent");
            p.refs.resize((size_t) count);
            for (auto& ref : p.refs) {
                if (!in.u64(ref.hash) || !in.u64(ref.hash2) || !in.u64(ref.bytes)) return false;
                if (ref.bytes != p.chunk) return in.fail("chunk size mismatch");
                manifest.entry.shared_bytes = sat_add(manifest.entry.shared_bytes, ref.bytes + kHeader + kTrailer);
            }
            if (!in.u64(tail) || tail != p.total - count * p.chunk ||
                tail > manifest.original_bytes - manifest.private_bytes)
                return in.fail("private tail size mismatch");
            manifest.private_bytes += tail;
        }
    }
    if (with_dflash2) {
        if (!in.u64(manifest.dflash2_bytes) || !manifest.dflash2_bytes ||
            manifest.dflash2_bytes > manifest.original_bytes - reconstructed)
            return in.fail("DFlash2 state exceeds logical session length");
        reconstructed += manifest.dflash2_bytes;
        manifest.private_bytes = sat_add(manifest.private_bytes, manifest.dflash2_bytes);
    }
    return reconstructed == manifest.original_bytes || in.fail("logical session length mismatch");
}

bool block_get_private(In& in, const BlockManifest& manifest, SavedConversation& image) {
    for (size_t checkpoint = 0; checkpoint < manifest.state_sizes.size(); ++checkpoint) {
        const auto parts = block_state_parts(checkpoint == 0 ? image.live : image.checkpoints[checkpoint - 1]);
        for (size_t part = 0; part < parts.size(); ++part) {
            parts[part]->resize((size_t) manifest.state_sizes[checkpoint][part]);
            if (!in.raw(parts[part]->data(), parts[part]->size())) return false;
        }
    }
    for (size_t layer = 0; layer < manifest.layers.size(); ++layer) {
        const auto parts = block_parts(image.kv[layer]);
        for (size_t part = 0; part < parts.size(); ++part) {
            const auto& p = manifest.layers[layer].parts[part];
            const size_t shared = (size_t) (p.chunk * p.refs.size());
            parts[part]->resize((size_t) p.total);
            if (!parts[part]->visit(shared, (size_t) p.total - shared,
                                   [&](uint8_t* data, size_t n, size_t) { return in.raw(data, n); })) return false;
        }
    }
    if (manifest.dflash2_bytes) {
        const uint64_t before = in.left;
        image.dflash2.emplace();
        if (!get_dflash2(in, *image.dflash2)) return false;
        if (before - in.left != manifest.dflash2_bytes) return in.fail("DFlash2 state extent mismatch");
    }
    return true;
}

bool block_load_manifest(const std::string& path, const SessionFileIdentity& id, const std::string* key,
                         BlockManifest& manifest, SavedConversation* image, const SessionReadLimits& limits,
                         std::string& error, SessionStatus& st) {
    RawFile raw;
    uint64_t size = 0;
    if (!raw.open_session(path, size))
        return block_fail(st, error, raw.error(), raw.refused() ? SessionError::invalid : raw.kind());
    if (size < kHeader || size > SIZE_MAX) return block_fail(st, error, "manifest size exceeds bounds");
    FileSource source(raw);
    if (!source.ok()) return block_fail(st, error, source.error(), SessionError::memory);
    if (limits.progress) source.progress.emplace(limits.progress);
    source.expected = size;
    if (!image) source.read_limit(kHeader);
    uint8_t header[kHeader], expected[kHeader];
    if (!source.read(header, sizeof header)) return block_fail(st, error, source.error(), source.kind());
    const uint64_t payload = load64(header + 32), metadata_bytes = load64(header + 40),
                   metadata_hash = load64(header + 48);
    uint32_t version = 0;
    std::memcpy(&version, header + 8, sizeof version);
    if (version != 2 && version != 3) return block_fail(st, error, "unsupported manifest version");
    block_header(expected, kManifestMagic, id, payload, metadata_bytes, metadata_hash, version);
    const uint64_t bound = sat_add(sat_mul(session_read_max_file_bytes(limits), 2), 1024);
    if (std::memcmp(header, expected, kHeader) || !metadata_bytes || metadata_bytes > payload ||
        metadata_bytes > size - kHeader || metadata_bytes > bound || payload > SIZE_MAX - kHeader - kTrailer)
        return block_fail(st, error, "invalid manifest header, identity or metadata extent");
    if (image && (size != payload + kHeader + kTrailer || size > bound))
        return block_fail(st, error, "private payload size mismatch");
    if (!image) source.read_limit(kHeader + metadata_bytes);
    In metadata{&source, SessionHasher(0), metadata_bytes, error, limits};
    if (!block_get_manifest(metadata, key, manifest, image, version == 3)) { st.error = metadata.kind; return false; }
    if (metadata.left || metadata.hash.digest() != metadata_hash)
        return block_fail(st, error, "manifest metadata checksum or extent mismatch");
    if (manifest.private_bytes != payload - metadata_bytes)
        return block_fail(st, error, "declared private payload extent mismatch");
    if (image) {
        In body{&source, metadata.hash, manifest.private_bytes, error, limits};
        if (!block_get_private(body, manifest, *image)) { st.error = body.kind; return false; }
        if (body.left) return block_fail(st, error, "unparsed private payload bytes");
        uint8_t trailer[kTrailer];
        if (!source.read(trailer, sizeof trailer)) return block_fail(st, error, source.error(), source.kind());
        if (load64(trailer) != body.hash.digest() || std::memcmp(trailer + 8, kEnd, 8))
            return block_fail(st, error, "private payload checksum or trailer mismatch");
    }
    manifest.entry.stored_bytes = size;
    manifest.entry.logical_bytes = sat_add(size, manifest.entry.shared_bytes);
    if (manifest_name(manifest.entry.key) != block_fs_path(path).filename().string())
        return block_fail(st, error, "manifest filename/key mismatch");
    // Preserve signed filesystem tick ordering in unsigned metadata; no duplicate persisted recency stamp.
    const auto stamp = fs::last_write_time(block_fs_path(path)).time_since_epoch().count();
    manifest.entry.used = uint64_t(stamp) ^ (uint64_t(1) << 63);
    st.error = SessionError::none;
    return true;
}

bool block_check_payload(const std::string& path, const SessionFileIdentity& id, const BlockRef& ref,
                         size_t offset, ConversationBuffer& target,
                         const SessionReadLimits& limits, std::string& error, SessionStatus& st) {
    uint64_t size = 0;
    return block_read_file(path, kChunkMagic, id, limits, sat_add(ref.bytes, kHeader + kTrailer),
        [&](In& in) {
            if (in.left != ref.bytes) return in.fail("chunk length mismatch");
            SessionHasher other(kContentSeed);
            uint8_t buf[65536];
            uint64_t at = 0;
            while (in.left) {
                const size_t n = (size_t) std::min<uint64_t>(in.left, sizeof buf);
                if (!in.raw(buf, n)) return false;
                other.update(buf, n);
                if (!target.visit(offset + (size_t) at, n,
                    [&](uint8_t* p, size_t count, size_t absolute) {
                        std::memcpy(p, buf + absolute - offset - (size_t) at, count); return true;
                    })) return in.fail("chunk target extent mismatch");
                at += n;
            }
            return (in.hash.digest() == ref.hash && other.digest() == ref.hash2) || in.fail("chunk content key mismatch");
        }, size, error, st);
}
// Reuse and collection need managed-file attribution; restoration verifies the full payload.
bool block_check_framing(const std::string& path, const SessionFileIdentity& id, const BlockRef& ref,
                         std::string& error, SessionStatus& st, const SessionWriteOptions* options = nullptr) {
    RawFile raw;
    uint64_t size = 0;
    if (!raw.open_session(path, size, options && options->durable))
        return block_fail(st, error, raw.error(), raw.refused() ? SessionError::invalid : raw.kind());
    if (ref.bytes > UINT64_MAX - kHeader - kTrailer || size != ref.bytes + kHeader + kTrailer)
        return block_fail(st, error, "managed chunk size mismatch");
    uint8_t header[kHeader], expected[kHeader], trailer[kTrailer];
    if (!raw.read_at(0, header, sizeof header) || !raw.read_at(size - kTrailer, trailer, sizeof trailer))
        return block_fail(st, error, raw.error(), raw.kind());
    block_header(expected, kChunkMagic, id, ref.bytes);
    if (std::memcmp(header, expected, kHeader) || load64(trailer) != ref.hash ||
        std::memcmp(trailer + 8, kEnd, 8))
        return block_fail(st, error, "invalid managed chunk framing or identity");
    // A previous nondurable write may have left the chunk's data only in the OS cache.
    if (options && options->durable) {
        if (options->phase) options->phase("flush", size);
        const int fault = options->fault ? options->fault("file_flush") : 0;
        if (fault) return block_fail(st, error, std::strerror(fault), errno_kind(fault));
        if (!raw.sync()) return block_fail(st, error, raw.error(), raw.kind());
    }
    return true;
}
BlockRef block_content(const ConversationBuffer& buffer, size_t offset, size_t n) {
    SessionHasher first, second(kContentSeed);
    buffer.visit(offset, n, [&](const uint8_t* p, size_t count, size_t) {
        first.update(p, count); second.update(p, count); return true;
    });
    return {first.digest(), second.digest(), n};
}
bool block_remove(const std::string& path, std::string& error, SessionStatus& st) {
    Publisher pub;
    if (!pub.open(path, error)) { st.error = pub.kind(); return false; }
    return pub.erase(error, st);
}
using BlockFiles = std::map<std::string, BlockRef>;
bool block_scan(const std::string& directory, const SessionFileIdentity& id, const SessionReadLimits& limits,
                std::vector<SessionBlockEntry>& entries, SessionBlockSummary& summary, BlockFiles& referenced,
                std::string& error, SessionStatus& st) {
    if (!block_dirs(directory, false, error, st)) return false;
    const fs::path dir = block_fs_path(directory);
    if (!fs::exists(dir / "manifests")) return true;
    uint64_t shared = 0;
    for (const auto& file : fs::directory_iterator(dir / "manifests")) {
        const auto name = file.path().filename().string();
        if (!named_hex_file(name, 2, ".manifest")) continue;
        BlockManifest manifest;
        if (!block_load_manifest(block_fs_string(file.path()), id, nullptr, manifest, nullptr, limits, error, st)) return false;
        for (const auto& layer : manifest.layers) for (const auto& part : layer.parts) for (const auto& ref : part.refs) {
            referenced.emplace(ref.name(), ref);
            shared = sat_add(shared, ref.bytes + kHeader + kTrailer);
        }
        summary.physical_bytes = sat_add(summary.physical_bytes, manifest.entry.stored_bytes);
        summary.logical_bytes = sat_add(summary.logical_bytes, manifest.entry.logical_bytes);
        entries.push_back(std::move(manifest.entry));
    }
    uint64_t unique = 0;
    for (const auto& [name, ref] : referenced) {
        RawFile raw;
        uint64_t size = 0;
        if (!raw.open_session(block_fs_string(dir / "blocks" / name), size))
            return block_fail(st, error, raw.error(), raw.refused() ? SessionError::invalid : raw.kind());
        if (size != ref.bytes + kHeader + kTrailer) return block_fail(st, error, "referenced chunk size mismatch");
        unique = sat_add(unique, size);
    }
    summary.entries = entries.size(); summary.blocks = referenced.size();
    summary.physical_bytes = sat_add(summary.physical_bytes, unique);
    summary.reused_bytes = shared - unique;
    for (const auto& file : fs::directory_iterator(dir / "blocks")) {
        const auto name = file.path().filename().string();
        if (!named_hex_file(name, 3, ".block") || referenced.count(name)) continue;
        BlockRef ref{std::stoull(name.substr(0, 16), nullptr, 16),
                     std::stoull(name.substr(17, 16), nullptr, 16),
                     std::stoull(name.substr(34, 16), nullptr, 16)};
        if (!block_check_framing(block_fs_string(file.path()), id, ref, error, st)) return false;
        summary.physical_bytes = sat_add(summary.physical_bytes, ref.bytes + kHeader + kTrailer);
        ++summary.blocks;
    }
    return true;
}
bool block_collect(const std::string& directory, const SessionFileIdentity& id, const BlockFiles& referenced,
                   std::string& error, SessionStatus& st, SessionBlockSummary* summary = nullptr) {
    const fs::path dir = block_fs_path(directory) / "blocks";
    if (!fs::exists(dir)) return true;
    // A previous publication/removal may have become visible but failed its directory flush. Make the entire
    // ownership snapshot durable before reclaiming any old-only chunk, even when this prune evicts no manifest.
    Publisher manifests;
    if (!manifests.open(block_fs_string(block_fs_path(directory) / "manifests" / ".ownership-barrier"), error)) {
        st.error = manifests.kind(); return false;
    }
    if (!manifests.sync_directory(error, st)) return false;
    for (const auto& file : fs::directory_iterator(dir)) {
        const auto name = file.path().filename().string();
        if (!named_hex_file(name, 3, ".block") || referenced.count(name)) continue;
        BlockRef ref;
        ref.hash = std::stoull(name.substr(0, 16), nullptr, 16);
        ref.hash2 = std::stoull(name.substr(17, 16), nullptr, 16);
        ref.bytes = std::stoull(name.substr(34, 16), nullptr, 16);
        // A name alone never authorizes deletion of an unrelated file.
        if (!block_check_framing(block_fs_string(file.path()), id, ref, error, st)) return false;
        if (!block_remove(block_fs_string(file.path()), error, st)) return false;
        if (summary) { summary->physical_bytes -= ref.bytes + kHeader + kTrailer; --summary->blocks; }
    }
    return true;
}
template<class Fn> bool block_guard(SessionStatus* status, std::string& error, Fn fn) {
    SessionStatus st;
    error.clear();
    bool ok = false;
    try { ok = fn(st); }
    catch (const std::bad_alloc&) { block_fail(st, error, "out of memory", SessionError::memory); }
    catch (const fs::filesystem_error& e) {
        block_fail(st, error, e.what(), errno_kind(e.code().value()));
    }
    if (ok) st.error = SessionError::none;
    if (status) *status = st;
    return ok;
}
} // namespace

SessionBlockStore::SessionBlockStore(std::string directory, SessionFileIdentity identity)
    : directory_(block_fs_string(block_fs_path(directory) / ("strata-session-blocks-v2-" + block_hex(identity.model) + "-" +
                                                            block_hex(identity.config)))), identity_(identity) {}
std::string SessionBlockStore::namespace_directory() const { return directory_; }

bool SessionBlockStore::write(const std::string& key, const SavedConversation& image, SessionBlockLayout layout,
                             size_t& bytes, std::string& error, const SessionWriteOptions& options, SessionStatus* status) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return block_guard(status, error, [&](SessionStatus& st) {
        if (key.empty() || image.live.ids.empty() || layout.main_kv_layers > image.kv.size() ||
            has_stage_parts(image) || !image.stage_images.empty())
            return block_fail(st, error, "empty key/state or unsupported capture shape");
        if (!block_dirs(directory_, true, error, st)) return false;
        const fs::path dir = block_fs_path(directory_);
        const std::string path = block_fs_string(dir / "manifests" / manifest_name(key));
        if (fs::exists(fs::symlink_status(block_fs_path(path)))) {
            BlockManifest old;
            if (!block_load_manifest(path, identity_, &key, old, nullptr, {}, error, st)) return false;
        }
        BlockManifest manifest;
        manifest.entry.layout = layout;
        Out original; put_payload(original, image);
        if (!original.ok || original.n > SIZE_MAX - kHeader - kTrailer)
            return block_fail(st, error, "session too large");
        manifest.original_bytes = original.n + kHeader + kTrailer;
        if (image.dflash2) {
            Out dflash2;
            put_dflash2(dflash2, *image.dflash2);
            if (!dflash2.ok) return block_fail(st, error, "DFlash2 state too large");
            manifest.dflash2_bytes = dflash2.n;
        }
        manifest.layers.resize(image.kv.size());
        bool unsupported_flush = false;
        for (size_t layer = 0; layer < image.kv.size(); ++layer) {
            if (image.kv[layer].cells < 0) return block_fail(st, error, "negative KV extent");
            const auto parts = block_parts(image.kv[layer]);
            for (size_t part = 0; part < 5; ++part) {
                auto& plan = manifest.layers[layer].parts[part];
                plan.total = parts[part]->size();
                plan.chunk = block_chunk_bytes(layer, part, plan.total, image.kv[layer].cells, layout);
                if (plan.chunk == UINT64_MAX) return block_fail(st, error, "KV part not cell aligned");
                const uint64_t chunks = plan.chunk ? block_whole_chunks(image.kv[layer].cells, image.live.ids.size()) : 0;
                for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
                    const size_t offset = (size_t) (chunk * plan.chunk);
                    const BlockRef ref = block_content(*parts[part], offset, (size_t) plan.chunk);
                    const std::string block_path = block_fs_string(dir / "blocks" / ref.name());
                    if (fs::exists(fs::symlink_status(block_fs_path(block_path)))) {
                        if (!block_check_framing(block_path, identity_, ref, error, st, &options))
                            return false;
                    } else {
                        uint64_t block_bytes = 0;
                        SessionStatus block_status;
                        if (!block_write_file(block_path, kChunkMagic, identity_, [&](Out& out) {
                            parts[part]->visit(offset, (size_t) ref.bytes,
                                [&](const uint8_t* p, size_t n, size_t) { out.raw(p, n); return out.ok; });
                        }, block_bytes, error, options, block_status)) {
                            st.error = block_status.error;
                            st.dir_flush_unsupported = block_status.dir_flush_unsupported;
                            return false; // a published block is not a published session
                        }
                        unsupported_flush = unsupported_flush || block_status.dir_flush_unsupported;
                    }
                    plan.refs.push_back(ref);
                    manifest.entry.shared_bytes = sat_add(manifest.entry.shared_bytes, ref.bytes + kHeader + kTrailer);
                }
            }
        }
        // Reused orphan names may originate in a write whose block-directory flush failed. Publish those names
        // durably now, before a new manifest can become a durable owner, even when this call wrote no new block.
        if (options.durable) {
            Publisher blocks;
            if (!blocks.open(block_fs_string(dir / "blocks" / ".publication-barrier"), error)) {
                st.error = blocks.kind(); return false;
            }
            if (!blocks.sync_directory(error, st)) return false;
        }
        SessionHasher metadata_hash;
        Out metadata; metadata.hash = &metadata_hash;
        block_put_metadata(metadata, key, image, manifest);
        if (!metadata.ok) return block_fail(st, error, "manifest metadata too large");
        uint64_t manifest_bytes = 0;
        if (!block_write_file(path, kManifestMagic, identity_,
            [&](Out& out) { block_put_metadata(out, key, image, manifest); block_put_private(out, image, manifest); },
            manifest_bytes, error, options, st, metadata.n, metadata_hash.digest(), image.dflash2 ? 3 : 2)) return false;
        st.dir_flush_unsupported = st.dir_flush_unsupported || unsupported_flush;
        bytes = (size_t) sat_add(manifest_bytes, manifest.entry.shared_bytes);
        return true;
    });
}
bool SessionBlockStore::read(const std::string& key, SavedConversation& image, size_t& bytes, std::string& error,
                            const SessionReadLimits& limits, SessionStatus* status) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return block_guard(status, error, [&](SessionStatus& st) {
        if (!block_dirs(directory_, false, error, st)) return false;
        const fs::path dir = block_fs_path(directory_);
        BlockManifest manifest;
        SavedConversation parsed;
        if (!block_load_manifest(block_fs_string(dir / "manifests" / manifest_name(key)), identity_, &key,
                                 manifest, &parsed, limits, error, st)) return false;
        for (size_t layer = 0; layer < manifest.layers.size(); ++layer)
            for (size_t part = 0; part < 5; ++part) {
                const auto& p = manifest.layers[layer].parts[part];
                for (size_t chunk = 0; chunk < p.refs.size(); ++chunk)
                    if (!block_check_payload(block_fs_string(dir / "blocks" / p.refs[chunk].name()), identity_, p.refs[chunk],
                        chunk * (size_t) p.chunk, *block_parts(parsed.kv[layer])[part], limits, error, st))
                        return false;
            }
        Out original; put_payload(original, parsed);
        if (!original.ok || original.n + kHeader + kTrailer != manifest.original_bytes)
            return block_fail(st, error, "reconstructed session length mismatch");
        image = std::move(parsed); bytes = (size_t) manifest.entry.logical_bytes; return true;
    });
}
bool SessionBlockStore::entries(std::vector<SessionBlockEntry>& entries, SessionBlockSummary& summary, std::string& error,
                               const SessionReadLimits& limits, SessionStatus* status) const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return block_guard(status, error, [&](SessionStatus& st) {
        std::vector<SessionBlockEntry> found; SessionBlockSummary stats; BlockFiles refs;
        if (!block_scan(directory_, identity_, limits, found, stats, refs, error, st)) return false;
        entries = std::move(found); summary = stats; return true;
    });
}
bool SessionBlockStore::erase(const std::string& key, std::string& error, SessionStatus* status) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return block_guard(status, error, [&](SessionStatus& st) {
        std::vector<SessionBlockEntry> found; SessionBlockSummary stats; BlockFiles refs;
        if (!block_scan(directory_, identity_, {}, found, stats, refs, error, st)) return false;
        const auto it = std::find_if(found.begin(), found.end(), [&](const auto& e) { return e.key == key; });
        if (it == found.end()) return true; // idempotent
        if (!block_remove(block_fs_string(block_fs_path(directory_) / "manifests" / manifest_name(key)), error, st)) return false;
        found.clear(); stats = {}; refs.clear();
        if (!block_scan(directory_, identity_, {}, found, stats, refs, error, st)) return false;
        return block_collect(directory_, identity_, refs, error, st);
    });
}
bool SessionBlockStore::touch(const std::string& key, std::string& error, SessionStatus* status) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return block_guard(status, error, [&](SessionStatus& st) {
        if (!block_dirs(directory_, false, error, st)) return false;
        const fs::path path = block_fs_path(directory_) / "manifests" / manifest_name(key);
        RawFile raw;
        uint64_t bytes = 0;
        if (!raw.open_session(block_fs_string(path), bytes))
            return block_fail(st, error, raw.error(), raw.refused() ? SessionError::invalid : raw.kind());
        if (!raw.close()) return block_fail(st, error, raw.error(), raw.kind());
        fs::last_write_time(path, fs::file_time_type::clock::now());
        return true;
    });
}
bool SessionBlockStore::prune(uint64_t max_bytes, size_t max_entries, SessionBlockSummary& summary, std::string& error,
                             const SessionReadLimits& limits, SessionStatus* status) {
    const std::lock_guard<std::mutex> lock(mutex_);
    return block_guard(status, error, [&](SessionStatus& st) {
        std::vector<SessionBlockEntry> found; SessionBlockSummary stats; BlockFiles refs;
        if (!block_scan(directory_, identity_, limits, found, stats, refs, error, st)) return false;
        if (!block_collect(directory_, identity_, refs, error, st, &stats)) return false;
        std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
            return a.used != b.used ? a.used < b.used : a.key < b.key;
        });
        while (stats.physical_bytes > max_bytes || stats.entries > max_entries) {
            if (found.empty()) break;
            if (!block_remove(block_fs_string(block_fs_path(directory_) / "manifests" / manifest_name(found.front().key)), error, st))
                return false;
            found.clear(); stats = {}; refs.clear();
            if (!block_scan(directory_, identity_, limits, found, stats, refs, error, st)) return false;
            if (!block_collect(directory_, identity_, refs, error, st, &stats)) return false;
            std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) {
                return a.used != b.used ? a.used < b.used : a.key < b.key;
            });
        }
        summary = stats; return true;
    });
}

} // namespace strata::core
