// SPDX-License-Identifier: BSD-2-Clause

// Copyright (c) 2019-2020, Paul Ferrand, Andrea Zanellato
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "FilePool.h"
#include "AudioReader.h"
#include "Buffer.h"
#include "AudioBuffer.h"
#include "AudioSpan.h"
#include "Config.h"
#include "utility/SwapAndPop.h"
#include "utility/Debug.h"
#include <ThreadPool.h>
#include <absl/types/span.h>
#include <absl/strings/match.h>
#include <absl/memory/memory.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <thread>
#include <system_error>
#include <atomic_queue/defs.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#endif
using namespace std::placeholders;

static std::weak_ptr<ThreadPool> globalThreadPoolWeakPtr;
static std::mutex globalThreadPoolMutex;

static unsigned decodeThreadCount()
{
    unsigned numThreads = std::thread::hardware_concurrency();
    return (numThreads > 2) ? (numThreads - 2) : 1;
}

// ---- Decode speed (organdroid patch, STREAMING_SAFETY_DESIGN.md §5) ----
// The app learns how fast this device decodes a pipe from what it already does: the
// parallel preload (head frames, the time each worker spent, the wall time) and every
// stream (frames decoded, the time its chunks took, opening included; and what sets how
// short a head can be: the first chunk's time, opening included, the wait from the voice's
// request to it, and how many streams were unfinished then). Worker
// threads write, the app reads; nothing here runs on the audio thread. Stream timings are
// kept in a small buffer the app drains; when it is full the oldest go.

namespace {
struct DecodeTiming { uint32_t frames, micros, firstMicros, waitMicros, ahead; };
constexpr size_t kMaxStreamTimings = 256;
std::mutex streamTimingsMutex;
std::vector<DecodeTiming> streamTimings;

std::atomic<uint64_t> preloadTimedFrames { 0 };
std::atomic<uint64_t> preloadThreadMicros { 0 };
std::atomic<uint64_t> preloadWallMicros { 0 };
std::atomic<int> preloadThreads { 0 };

uint64_t microsSince(std::chrono::steady_clock::time_point start)
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count());
}

uint32_t clamp32(uint64_t v) { return static_cast<uint32_t>(std::min<uint64_t>(v, UINT32_MAX)); }

void recordStreamTiming(uint64_t frames, uint64_t micros, uint64_t firstMicros, uint64_t waitMicros, int ahead)
{
    std::lock_guard<std::mutex> lock { streamTimingsMutex };
    if (streamTimings.size() >= kMaxStreamTimings)
        streamTimings.erase(streamTimings.begin());
    streamTimings.push_back({ clamp32(frames), clamp32(micros), clamp32(firstMicros), clamp32(waitMicros),
                              static_cast<uint32_t>(std::max(ahead, 0)) });
}
} // namespace

// The last cold preload: head frames decoded, worker time summed over jobs, wall time,
// workers. Zero until a preload decoded something.
extern "C" void sfizz_preload_timing(uint64_t* frames, uint64_t* threadMicros, uint64_t* wallMicros, int* threads)
{
    if (frames)       *frames       = preloadTimedFrames.load(std::memory_order_relaxed);
    if (threadMicros) *threadMicros = preloadThreadMicros.load(std::memory_order_relaxed);
    if (wallMicros)   *wallMicros   = preloadWallMicros.load(std::memory_order_relaxed);
    if (threads)      *threads      = preloadThreads.load(std::memory_order_relaxed);
}

// Moves up to max stream timings (oldest first) into the arrays; returns how many.
extern "C" int sfizz_stream_timings(uint32_t* frames, uint32_t* micros, uint32_t* firstMicros,
                                    uint32_t* waitMicros, uint32_t* ahead, int max)
{
    std::lock_guard<std::mutex> lock { streamTimingsMutex };
    const int n = std::min<int>(max, static_cast<int>(streamTimings.size()));
    for (int i = 0; i < n; ++i) {
        frames[i] = streamTimings[i].frames;
        micros[i] = streamTimings[i].micros;
        firstMicros[i] = streamTimings[i].firstMicros;
        waitMicros[i] = streamTimings[i].waitMicros;
        ahead[i] = streamTimings[i].ahead;
    }
    streamTimings.erase(streamTimings.begin(), streamTimings.begin() + n);
    return n;
}

// The streamed buffers held in RAM and the collector's last pass (design 4f; the M31 check
// found streamed files kept after playing). Bytes and files are added when a stream is
// Done and taken off when the collector frees it; reset at each palette load (a palette's
// own pool is freed whole with it). The pass figures are the last pass's: files kept
// because a voice still reads them, kept because idle less than fileClearingPeriod, and
// freed, plus the lastUsedFiles left after it.
namespace {
std::atomic<int64_t> streamBytes { 0 };
std::atomic<int> streamFiles { 0 };
std::atomic<int> gcPasses { 0 };
std::atomic<int> gcKeptReading { 0 };
std::atomic<int> gcKeptRecent { 0 };
std::atomic<int> gcFreed { 0 };
std::atomic<int> gcPending { 0 };
std::atomic<int> gcDue { 0 };          // passes the Synth asked for
std::atomic<int> gcLockMissed { 0 };   // of those, skipped: the collector's lock was held

int64_t streamedBytes(const sfz::FileData& d)
{
    return static_cast<int64_t>(d.fileData.getNumFrames() * d.fileData.getNumChannels() * sizeof(float)
        + d.packedTail.getNumFrames() * d.packedTail.getNumChannels() * sizeof(int16_t));
}
} // namespace

// out: {bytes, files, passes, kept reading, kept recent, freed, pending, due, lock missed}
extern "C" void sfizz_stream_cache(int64_t* out)
{
    out[0] = streamBytes.load(std::memory_order_relaxed);
    out[1] = streamFiles.load(std::memory_order_relaxed);
    out[2] = gcPasses.load(std::memory_order_relaxed);
    out[3] = gcKeptReading.load(std::memory_order_relaxed);
    out[4] = gcKeptRecent.load(std::memory_order_relaxed);
    out[5] = gcFreed.load(std::memory_order_relaxed);
    out[6] = gcPending.load(std::memory_order_relaxed);
    out[7] = gcDue.load(std::memory_order_relaxed);
    out[8] = gcLockMissed.load(std::memory_order_relaxed);
}

// The cache budget (design 4f): released streamed files are kept while their bytes fit in
// it, and over it the ones idle longest are freed first. -1 = sfizz's rule (free what has
// been idle fileClearingPeriod). Set by the app from what the heads leave spare; lowered on
// onTrimMemory, which also asks for a pass at once.
namespace {
std::atomic<int64_t> cacheBudgetBytes { -1 };
std::atomic<bool> collectRequested { false };
} // namespace

extern "C" void sfizz_set_stream_cache_budget(int64_t bytes)
{
    cacheBudgetBytes.store(bytes < 0 ? -1 : bytes, std::memory_order_relaxed);
}

extern "C" void sfizz_collect_streams_now()
{
    collectRequested.store(true, std::memory_order_relaxed);
}

// Late requests (design 4e, 9b): the app sets the latest safe moment at every load, its heads
// less their floor (HeadPlan.kt); 0, at the voice's start as sfizz did, until it does.
namespace { std::atomic<float> streamRequestDelaySeconds { 0.0f }; }
extern "C" void sfizz_set_stream_request_delay(float seconds)
{
    streamRequestDelaySeconds.store(seconds > 0.0f ? seconds : 0.0f, std::memory_order_relaxed);
}
float sfz::streamRequestDelay() noexcept
{
    return streamRequestDelaySeconds.load(std::memory_order_relaxed);
}

namespace {
std::atomic<bool> earlyTailsFlag { true };
std::atomic<int> tailsAsked { 0 };     // tail streams queued at a note-on
std::atomic<int> tailsMovedUp { 0 };   // of those, still waiting when their tail started
} // namespace
extern "C" void sfizz_set_early_tails(int on)
{
    earlyTailsFlag.store(on != 0, std::memory_order_relaxed);
}
extern "C" void sfizz_tail_stream_counts(int* asked, int* movedUp)
{
    if (asked) *asked = tailsAsked.load();
    if (movedUp) *movedUp = tailsMovedUp.load();
}
bool sfz::earlyTails() noexcept
{
    return earlyTailsFlag.load(std::memory_order_relaxed);
}

// Test only (tools/composite_bench/srender late:): while on, every voice reads its head as
// if no stream had arrived, so the never-silent fallback (design 4d) can be rendered.
namespace { std::atomic<bool> streamsWithheldFlag { false }; }
extern "C" void sfizz_set_streams_withheld(int on)
{
    streamsWithheldFlag.store(on != 0, std::memory_order_relaxed);
}
bool sfz::streamsWithheld() noexcept
{
    return streamsWithheldFlag.load(std::memory_order_relaxed);
}

bool sfz::FilePool::garbageCollectionDue(size_t framesSince, double sampleRate) noexcept
{
    if (collectRequested.exchange(false, std::memory_order_relaxed))
        return true;
    // With a budget the time no longer decides what is freed, so pass every second: a
    // file over the budget waits at most that.
    const double period = cacheBudgetBytes.load(std::memory_order_relaxed) < 0
        ? double(config::fileClearingPeriod) : 1.0;
    return double(framesSince) > period * sampleRate;
}

extern "C" void sfizz_stream_cache_reset()
{
    streamBytes.store(0, std::memory_order_relaxed);
    streamFiles.store(0, std::memory_order_relaxed);
    gcPasses.store(0, std::memory_order_relaxed);
    gcDue.store(0, std::memory_order_relaxed);
    gcLockMissed.store(0, std::memory_order_relaxed);
}

// Threads that serve streams (and decode the preload): the same count for both.
extern "C" int sfizz_decode_threads()
{
    return static_cast<int>(decodeThreadCount());
}

static std::shared_ptr<ThreadPool> globalThreadPool()
{
    std::shared_ptr<ThreadPool> threadPool;

    threadPool = globalThreadPoolWeakPtr.lock();
    if (threadPool)
        return threadPool;

    std::lock_guard<std::mutex> lock(globalThreadPoolMutex);
    threadPool = globalThreadPoolWeakPtr.lock();
    if (threadPool)
        return threadPool;

    threadPool.reset(new ThreadPool(decodeThreadCount()));
    globalThreadPoolWeakPtr = threadPool;
    return threadPool;
}

void readBaseFile(sfz::AudioReader& reader, sfz::FileAudioBuffer& output, uint32_t numFrames)
{
    output.reset();
    output.resize(numFrames);

    const unsigned channels = reader.channels();

    if (channels == 1) {
        output.addChannel();
        output.clear();
        reader.readNextBlock(output.channelWriter(0), numFrames);
    } else if (channels == 2) {
        output.addChannel();
        output.addChannel();
        output.clear();
        sfz::Buffer<float> tempReadBuffer { 2 * numFrames };
        reader.readNextBlock(tempReadBuffer.data(), numFrames);
        sfz::readInterleaved(tempReadBuffer, output.getSpan(0), output.getSpan(1));
    }
}

sfz::FileAudioBuffer readFromFile(sfz::AudioReader& reader, uint32_t numFrames)
{
    sfz::FileAudioBuffer baseBuffer;
    readBaseFile(reader, baseBuffer, numFrames);
    return baseBuffer;
}

// The 16-bit values of a float head, or false when a sample is not one (a 24-bit or float
// recording): then the head stays float, as before. Multiplying by 32768 is exact in
// float, so the test is exact, and so is the way back (kPackedHeadScale).
static bool packHead(const sfz::FileAudioBuffer& head, sfz::PackedHeadBuffer& out)
{
    const size_t numFrames = head.getNumFrames();
    const size_t numChannels = head.getNumChannels();
    if (numFrames == 0 || numChannels == 0)
        return false;
    for (size_t c = 0; c < numChannels; ++c) {
        for (float v : head.getConstSpan(c)) {
            const float k = v * 32768.0f;
            if (!(k >= -32768.0f && k <= 32767.0f) || k != std::nearbyint(k))
                return false;
        }
    }
    out.reset();
    out.resize(numFrames);
    for (size_t c = 0; c < numChannels; ++c)
        out.addChannel();
    out.clear();
    for (size_t c = 0; c < numChannels; ++c) {
        const auto from = head.getConstSpan(c);
        const auto to = out.getSpan(c);
        for (size_t i = 0; i < numFrames; ++i)
            to[i] = static_cast<int16_t>(from[i] * 32768.0f);
    }
    return true;
}

// On by default. Off keeps every head float, as before: the A/B for the null test and the
// render cost. Read when a head is set, so a change applies from the next load.
static std::atomic<bool> packHeads { true };
extern "C" void sfizz_set_packed_heads(int on)
{
    packHeads.store(on != 0, std::memory_order_relaxed);
}

// Heads set since the last reset, packed and kept float: the check that packing happens
// (a null test passes trivially if it never does) and the log line of each load.
static std::atomic<int> headsPacked { 0 };
static std::atomic<int> headsFloat { 0 };
extern "C" void sfizz_head_counts(int* packed, int* floats)
{
    if (packed) *packed = headsPacked.load(std::memory_order_relaxed);
    if (floats) *floats = headsFloat.load(std::memory_order_relaxed);
}
extern "C" void sfizz_head_counts_reset()
{
    headsPacked.store(0, std::memory_order_relaxed);
    headsFloat.store(0, std::memory_order_relaxed);
}

void sfz::FileData::setHead(FileAudioBuffer&& head) noexcept
{
    PackedHeadBuffer packed;
    if (packHeads.load(std::memory_order_relaxed) && !information.wavetable && packHead(head, packed)) {
        packedHead = std::move(packed);
        preloadedData.reset();
        headsPacked.fetch_add(1, std::memory_order_relaxed);
    } else {
        packedHead.reset();
        preloadedData = std::move(head);
        headsFloat.fetch_add(1, std::memory_order_relaxed);
    }
}

sfz::FilePool::FilePool()
    : filesToLoad(alignedNew<FileQueue>()),
      threadPool(globalThreadPool())
{
    activeStreams.reserve(config::maxVoices);
    lastUsedFiles.reserve(config::maxVoices);
    gcCandidates.reserve(config::maxVoices);
    garbageToCollect.reserve(config::maxVoices);
    packedGarbage.reserve(config::maxVoices);
}

sfz::FilePool::~FilePool()
{
    std::error_code ec;

    garbageFlag = false;
    semGarbageBarrier.post(ec);
    garbageThread.join();

    dispatchFlag = false;
    dispatchBarrier.post(ec);
    dispatchThread.join();

    // Every task holds `this`: cancel the streams and wait until the last task is gone.
    streamsCancelled = true;
    while (streamTasks.load() > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
}

bool sfz::FilePool::checkSample(std::string& filename) const noexcept
{
    fs::path path { rootDirectory / filename };
    std::error_code ec;
    if (fs::exists(path, ec))
        return true;

#if defined(_WIN32)
    return false;
#else
    fs::path oldPath = std::move(path);
    path = oldPath.root_path();

    static const fs::path dot { "." };
    static const fs::path dotdot { ".." };

    for (const fs::path& part : oldPath.relative_path()) {
        if (part == dot || part == dotdot) {
            path /= part;
            continue;
        }

        if (fs::exists(path / part, ec)) {
            path /= part;
            continue;
        }

        auto it = path.empty() ? fs::directory_iterator { dot, ec } : fs::directory_iterator { path, ec };
        if (ec) {
            DBG("Error creating a directory iterator for " << filename << " (Error code: " << ec.message() << ")");
            return false;
        }

        auto searchPredicate = [&part](const fs::directory_entry &ent) -> bool {
#if !defined(GHC_USE_WCHAR_T)
            return absl::EqualsIgnoreCase(
                ent.path().filename().native(), part.native());
#else
            return absl::EqualsIgnoreCase(
                ent.path().filename().u8string(), part.u8string());
#endif
        };

        while (it != fs::directory_iterator {} && !searchPredicate(*it))
            it.increment(ec);

        if (it == fs::directory_iterator {}) {
            DBG("File not found, could not resolve " << filename);
            return false;
        }

        path /= it->path().filename();
    }

    const auto newPath = fs::relative(path, rootDirectory, ec);
    if (ec) {
        DBG("Error extracting the new relative path for " << filename << " (Error code: " << ec.message() << ")");
        return false;
    }
    DBG("Updating " << filename << " to " << newPath);
    filename = newPath.string();
    return true;
#endif
}

bool sfz::FilePool::checkSampleId(FileId& fileId) const noexcept
{
    if (loadedFiles.contains(fileId))
        return true;

    std::string filename = fileId.filename();
    bool result = checkSample(filename);
    if (result)
        fileId = FileId(std::move(filename), fileId.isReverse());
    return result;
}

absl::optional<sfz::FileInformation> getReaderInformation(sfz::AudioReader* reader) noexcept
{
    const unsigned channels = reader->channels();
    if (channels != 1 && channels != 2)
        return {};

    sfz::FileInformation returnedValue;
    returnedValue.end = static_cast<uint32_t>(reader->frames()) - 1;
    returnedValue.sampleRate = static_cast<double>(reader->sampleRate());
    returnedValue.numChannels = static_cast<int>(channels);

    // Check for instrument info
    sfz::InstrumentInfo instrumentInfo {};
    if (reader->getInstrumentInfo(instrumentInfo)) {
        returnedValue.rootKey = clamp<uint8_t>(instrumentInfo.basenote, 0, 127);
        if (reader->type() == sfz::AudioReaderType::Forward) {
            if (instrumentInfo.loop_count > 0) {
                returnedValue.hasLoop = true;
                returnedValue.loopStart = instrumentInfo.loops[0].start;
                returnedValue.loopEnd =
                    min(returnedValue.end, static_cast<int64_t>(instrumentInfo.loops[0].end - 1));
            }
        } else {
            // TODO loops ignored when reversed
            //   prehaps it can make use of SF_LOOP_BACKWARD?
        }
    }

    // Check for wavetable info
    sfz::WavetableInfo wt {};
    if (reader->getWavetableInfo(wt))
        returnedValue.wavetable = wt;

    return returnedValue;
}

absl::optional<sfz::FileInformation> sfz::FilePool::checkExistingFileInformation(const FileId& fileId) noexcept
{
    const auto loadedFile = loadedFiles.find(fileId);
    if (loadedFile != loadedFiles.end())
        return loadedFile->second.information;

    const auto preloadedFile = preloadedFiles.find(fileId);
    if (preloadedFile != preloadedFiles.end())
        return preloadedFile->second.information;

    return {};
}

absl::optional<sfz::FileInformation> sfz::FilePool::getFileInformation(const FileId& fileId) noexcept
{
    auto existingInformation = checkExistingFileInformation(fileId);
    if (existingInformation)
        return existingInformation;

    const fs::path file { rootDirectory / fileId.filename() };

    if (!fs::exists(file))
        return {};

    AudioReaderPtr reader = createAudioReader(file, fileId.isReverse());
    return getReaderInformation(reader.get());
}

bool sfz::FilePool::preloadFile(const FileId& fileId, uint32_t maxOffset, uint32_t headFrames) noexcept
{
    const auto loadedFile = loadedFiles.find(fileId);
    if (loadedFile != loadedFiles.end()) {
        loadedFile->second.preloadCallCount++;
        return true;
    }

    auto fileInformation = getFileInformation(fileId);
    if (!fileInformation)
        return false;

    fileInformation->maxOffset = maxOffset;
    fileInformation->headFrames = headFrames;
    const fs::path file { rootDirectory / fileId.filename() };
    AudioReaderPtr reader = createAudioReader(file, fileId.isReverse());

    const auto frames = static_cast<uint32_t>(reader->frames());
    const auto framesToLoad = [&]() {
        if (loadInRam)
            return frames;
        else
            return min(frames, maxOffset + (headFrames > 0 ? headFrames : preloadSize));
    }();

    const auto existingFile = preloadedFiles.find(fileId);
    if (existingFile != preloadedFiles.end()) {
        auto& fileData = existingFile->second;
        if (framesToLoad > fileData.headFrames()) {
            fileData.information.maxOffset = maxOffset;
            fileData.information.headFrames = headFrames;
            fileData.setHead(readFromFile(*reader, framesToLoad));
            fileData.fullyLoaded = frames == framesToLoad;
        }
        fileData.preloadCallCount++;
    } else {
        fileInformation->sampleRate = static_cast<double>(reader->sampleRate());
        auto insertedPair = preloadedFiles.insert_or_assign(fileId, {
            FileAudioBuffer {},
            *fileInformation
        });
        insertedPair.first->second.setHead(readFromFile(*reader, framesToLoad));

        insertedPair.first->second.preloadCallCount++;
        insertedPair.first->second.status = FileData::Status::Preloaded;
        insertedPair.first->second.fullyLoaded = framesToLoad == frames;
    }

    return true;
}

// ---- Batch preload with parallel decode (organdroid patch) ----
// Cold-loading a large SFZ preloads the head of a few thousand FLAC files; doing that
// sequentially (one preloadFile per region file) leaves every core but one idle and
// dominates load time. preloadFiles() decodes fresh files on a short-lived local pool,
// then publishes the results into preloadedFiles sequentially, so no shared FilePool
// state is ever touched from a worker. The done/total atomics feed the host app's
// loading bar; they are process-global (one synth loads at a time in the organ app).

static std::atomic<int> preloadProgressDone { 0 };
static std::atomic<int> preloadProgressTotal { 0 };

extern "C" void sfizz_preload_progress(int* done, int* total)
{
    if (done)  *done  = preloadProgressDone.load(std::memory_order_relaxed);
    if (total) *total = preloadProgressTotal.load(std::memory_order_relaxed);
}

extern "C" void sfizz_preload_progress_reset()
{
    preloadProgressDone.store(0, std::memory_order_relaxed);
    preloadProgressTotal.store(0, std::memory_order_relaxed);
}

void sfz::FilePool::preloadFiles(const std::vector<PreloadRequest>& files) noexcept
{
    preloadProgressDone.store(0, std::memory_order_relaxed);
    preloadProgressTotal.store(static_cast<int>(files.size()), std::memory_order_relaxed);

    // Files already tracked keep preloadFile()'s exact sequential bookkeeping; only
    // files in neither map (the whole set on a cold load) are farmed out to workers.
    struct PreloadJob {
        const FileId* id;
        uint32_t maxOffset;
        uint32_t headFrames;
        absl::optional<FileInformation> information;
        FileData data;          // the head set in the worker (setHead packs it there)
        bool fullyLoaded { false };
        bool ok { false };
    };
    std::vector<PreloadJob> jobs;
    jobs.reserve(files.size());
    for (const auto& item : files) {
        if (loadedFiles.contains(item.id) || preloadedFiles.contains(item.id)) {
            preloadFile(item.id, item.maxOffset, item.headFrames);
            preloadProgressDone.fetch_add(1, std::memory_order_relaxed);
        } else {
            jobs.emplace_back();
            jobs.back().id = &item.id;
            jobs.back().maxOffset = item.maxOffset;
            jobs.back().headFrames = item.headFrames;
        }
    }

    if (!jobs.empty()) {
        // A dedicated pool, NOT the shared streaming pool: a still-audible synth (the old
        // palette during a hot-swap) streams through that pool, and its jobs must not
        // queue behind thousands of decode jobs.
        const unsigned numThreads = decodeThreadCount();
        ThreadPool pool { numThreads };
        std::vector<std::future<void>> futures;
        futures.reserve(jobs.size());
        std::atomic<uint64_t> framesDecoded { 0 };
        std::atomic<uint64_t> workerMicros { 0 };
        const auto wallStart = std::chrono::steady_clock::now();
        for (PreloadJob& job : jobs) {
            futures.push_back(pool.enqueue([this, &job, &framesDecoded, &workerMicros] {
                const auto jobStart = std::chrono::steady_clock::now();
                const fs::path file { rootDirectory / job.id->filename() };
                std::error_code ec;
                if (fs::exists(file, ec)) {
                    AudioReaderPtr reader = createAudioReader(file, job.id->isReverse());
                    auto information = getReaderInformation(reader.get());
                    if (information) {
                        information->maxOffset = job.maxOffset;
                        information->headFrames = job.headFrames;
                        information->sampleRate = static_cast<double>(reader->sampleRate());
                        const auto frames = static_cast<uint32_t>(reader->frames());
                        const auto framesToLoad = loadInRam ? frames
                            : min(frames, job.maxOffset + (job.headFrames > 0 ? job.headFrames : preloadSize));
                        job.data.information = *information;
                        job.data.setHead(readFromFile(*reader, framesToLoad));
                        job.fullyLoaded = framesToLoad == frames;
                        job.information = information;
                        job.ok = true;
                        framesDecoded.fetch_add(framesToLoad, std::memory_order_relaxed);
                    }
                }
                workerMicros.fetch_add(microsSince(jobStart), std::memory_order_relaxed);
                preloadProgressDone.fetch_add(1, std::memory_order_relaxed);
            }));
        }
        for (auto& f : futures)
            f.wait();
        // Only a preload that decoded files replaces the last timing (a reload of files
        // already held decodes nothing).
        if (framesDecoded.load() > 0) {
            preloadTimedFrames.store(framesDecoded.load(), std::memory_order_relaxed);
            preloadThreadMicros.store(workerMicros.load(), std::memory_order_relaxed);
            preloadWallMicros.store(microsSince(wallStart), std::memory_order_relaxed);
            preloadThreads.store(static_cast<int>(numThreads), std::memory_order_relaxed);
        }
    }

    for (PreloadJob& job : jobs) {
        if (!job.ok)
            continue;
        auto insertedPair = preloadedFiles.insert_or_assign(*job.id, std::move(job.data));
        insertedPair.first->second.preloadCallCount++;
        insertedPair.first->second.status = FileData::Status::Preloaded;
        insertedPair.first->second.fullyLoaded = job.fullyLoaded;
    }
}

void sfz::FilePool::resetPreloadCallCounts() noexcept
{
    for (auto& preloadedFile: preloadedFiles)
        preloadedFile.second.preloadCallCount = 0;

    for (auto& loadedFile: loadedFiles)
        loadedFile.second.preloadCallCount = 0;
}

void sfz::FilePool::removeUnusedPreloadedData() noexcept
{
    for (auto it = preloadedFiles.begin(), end = preloadedFiles.end(); it != end; ) {
        auto copyIt = it++;
        if (copyIt->second.preloadCallCount == 0) {
            DBG("[sfizz] Removing unused preloaded data: " << copyIt->first.filename());
            preloadedFiles.erase(copyIt);
        }
    }

    for (auto it = loadedFiles.begin(), end = loadedFiles.end(); it != end; ) {
        auto copyIt = it++;
        if (copyIt->second.preloadCallCount == 0) {
            DBG("[sfizz] Removing unused loaded data: " << copyIt->first.filename());
            loadedFiles.erase(copyIt);
        }
    }
}

sfz::FileDataHolder sfz::FilePool::loadFile(const FileId& fileId) noexcept
{
    auto fileInformation = getFileInformation(fileId);
    if (!fileInformation)
        return {};

    const auto existingFile = loadedFiles.find(fileId);
    if (existingFile != loadedFiles.end()) {
        existingFile->second.preloadCallCount++;
        return { &existingFile->second };
    }

    const fs::path file { rootDirectory / fileId.filename() };
    AudioReaderPtr reader = createAudioReader(file, fileId.isReverse());

    const auto frames = static_cast<uint32_t>(reader->frames());
    auto insertedPair = loadedFiles.insert_or_assign(fileId, {
        readFromFile(*reader, frames),
        *fileInformation
    });
    insertedPair.first->second.preloadCallCount++;
    insertedPair.first->second.status = FileData::Status::Preloaded;
    insertedPair.first->second.fullyLoaded = true;
    return { &insertedPair.first->second };
}

sfz::FileDataHolder sfz::FilePool::loadFromRam(const FileId& fileId, const std::vector<char>& data) noexcept
{
    const auto loaded = loadedFiles.find(fileId);
    if (loaded != loadedFiles.end())
        return { &loaded->second };

    auto reader = createAudioReaderFromMemory(data.data(), data.size(), fileId.isReverse());
    auto fileInformation = getReaderInformation(reader.get());
    const auto frames = static_cast<uint32_t>(reader->frames());
    auto insertedPair = loadedFiles.insert_or_assign(fileId, {
        readFromFile(*reader, frames),
        *fileInformation
    });
    insertedPair.first->second.preloadCallCount++;
    insertedPair.first->second.status = FileData::Status::Preloaded;
    insertedPair.first->second.fullyLoaded = true;
    DBG("Added a file " << fileId.filename());
    return { &insertedPair.first->second };
}

sfz::FileDataHolder sfz::FilePool::getFilePromise(const std::shared_ptr<FileId>& fileId, bool stream) noexcept
{
    const auto loaded = loadedFiles.find(*fileId);
    if (loaded != loadedFiles.end())
        return { &loaded->second };

    const auto preloaded = preloadedFiles.find(*fileId);
    if (preloaded == preloadedFiles.end()) {
        DBG("[sfizz] File not found in the preloaded files: " << fileId->filename());
        return {};
    }

    if (stream)
        queueStream(fileId, preloaded->second, streamClock());
    return { &preloaded->second };
}

void sfz::FilePool::requestStream(const std::shared_ptr<FileId>& fileId, double sounded, int64_t reach) noexcept
{
    if (loadedFiles.find(*fileId) != loadedFiles.end())
        return;
    const auto preloaded = preloadedFiles.find(*fileId);
    if (preloaded == preloadedFiles.end())
        return;
    if (reach >= 0 && reach <= static_cast<int64_t>(preloaded->second.headFrames()))
        return;
    queueStream(fileId, preloaded->second, streamClock() - sounded);
}

void sfz::FilePool::requestTail(const std::shared_ptr<FileId>& fileId, int64_t reach) noexcept
{
    if (loadedFiles.find(*fileId) != loadedFiles.end())
        return;
    const auto preloaded = preloadedFiles.find(*fileId);
    if (preloaded == preloadedFiles.end())
        return;
    if (reach >= 0 && reach <= static_cast<int64_t>(preloaded->second.headFrames()))
        return;
    queueStream(fileId, preloaded->second, streamClock(), true);
}

void sfz::FilePool::promoteTail(const std::shared_ptr<FileId>& fileId, double sounded) noexcept
{
    const auto preloaded = preloadedFiles.find(*fileId);
    if (preloaded != preloadedFiles.end() && preloaded->second.tailPending.load())
        queueStream(fileId, preloaded->second, streamClock() - sounded);
}

void sfz::FilePool::queueStream(const std::shared_ptr<FileId>& fileId, FileData& fileData, double startedAt, bool tail) noexcept
{
    // organdroid: queue the stream only when nobody has yet. An organ chord strikes the
    // same files over and over (every stop x every key, then their release tails), and a
    // stream for a file already Streaming/Done is dropped at once (startStream), so those pushes
    // only filled the 256-slot queue; a full queue then refused the voice outright.
    // One exception (design 9d): a request with a deadline for a file asked as a certain
    // tail goes through, once, for the dispatcher to move that stream up.
    const auto status = fileData.status.load();
    const bool needsStream = status == FileData::Status::Preloaded
                          || status == FileData::Status::GarbageCollecting;
    const bool promote = !tail && fileData.tailPending.exchange(false)
                      && status == FileData::Status::Streaming;
    if (fileData.fullyLoaded || !(needsStream || promote))
        return;
    QueuedFileData queuedData { fileId, &fileData, streamClock(), startedAt, tail };
    if (tail)
        fileData.tailPending = true;
    if (!filesToLoad->try_push(queuedData)) {
        if (tail)
            fileData.tailPending = false;
        // Still full: play from the preloaded head rather than dropping the note. The
        // voice ends early only if it outruns the head before a later voice queues
        // the stream.
        DBG("[sfizz] Could not enqueue the file to load for " << fileId << " (queue capacity " << filesToLoad->capacity() << ")");
        return;
    }

    pendingStreams.fetch_add(1);
    if (tail)
        tailsAsked.fetch_add(1, std::memory_order_relaxed);
    std::error_code ec;
    dispatchBarrier.post(ec);
    ASSERT(!ec);
}

void sfz::FilePool::setPreloadSize(uint32_t preloadSize) noexcept
{
    this->preloadSize = preloadSize;
    if (loadInRam)
        return;

    // Update all the preloaded sizes
    for (auto& preloadedFile : preloadedFiles) {
        auto& fileId = preloadedFile.first;
        auto& fileData = preloadedFile.second;
        // organdroid: one size for every file, the per-stop heads included (the app's
        // preload probe; a palette load stamps the heads again).
        fileData.information.headFrames = 0;
        const auto maxOffset = fileData.information.maxOffset;
        fs::path file { rootDirectory / fileId.filename() };
        AudioReaderPtr reader = createAudioReader(file, fileId.isReverse());
        const auto frames = reader->frames();
        const auto framesToLoad = min(frames, maxOffset + preloadSize);
        fileData.setHead(readFromFile(*reader, static_cast<uint32_t>(framesToLoad)));
        fileData.fullyLoaded = frames == framesToLoad;
    }
}

// ---- Streaming (organdroid, STREAMING_SAFETY_DESIGN.md 4a) ----
// sfizz decoded each file whole, from frame 0, one job per file, first in first out: a file
// queued behind others waited for all of them to be decoded to their ends, and a burst of
// stops queued many at once. Now a stream starts where the preloaded head ends (the head
// is copied in, not decoded again) and is decoded kStreamChunkFrames at a time; each task
// takes the stream whose deadline is soonest, the moment its earliest voice reaches the
// end of what is decoded (taken as playing at the file's own rate from the request on).
// A pipe then waits for chunks, not for whole files.

namespace {
// 170 ms of a 48 kHz pipe: about 1 ms of decoding on the Galaxy M31 (7-9 M frames/s per
// thread, tests/STREAMING_PHONE_CHECK_REPORT.md), so an urgent stream waits about that per
// task ahead of it; the requeue costs microseconds.
constexpr size_t kStreamChunkFrames { 8192 };

// Off by default. On, every finished stream is compared with the file decoded whole from
// frame 0 (the old way) and counted: the check that the copied head and the seek are exact.
std::atomic<bool> verifyStreams { false };
std::atomic<int> streamsVerified { 0 };
std::atomic<int> streamsMismatched { 0 };
// Streams started after a copied head, and from frame 0 (a reader that cannot seek).
std::atomic<int> streamsFromHead { 0 };
std::atomic<int> streamsFromZero { 0 };

// On by default. A stream of a 16-bit file after a packed head keeps only the frames past
// the head, packed as the head is: half the float copy per frame, and nothing for the
// head. Off streams every file as floats, as before: the A/B for the null test.
std::atomic<bool> packStreams { true };
std::atomic<int> streamsPacked { 0 };
} // namespace

extern "C" void sfizz_set_packed_streams(int on)
{
    packStreams.store(on != 0);
}

// Streams started with a packed tail since the start.
extern "C" int sfizz_streams_packed()
{
    return streamsPacked.load();
}

extern "C" void sfizz_stream_starts(int* fromHead, int* fromZero)
{
    if (fromHead) *fromHead = streamsFromHead.load();
    if (fromZero) *fromZero = streamsFromZero.load();
}

extern "C" void sfizz_set_stream_verify(int on)
{
    verifyStreams.store(on != 0);
}

extern "C" void sfizz_stream_verify_counts(int* ok, int* bad)
{
    if (ok) *ok = streamsVerified.load();
    if (bad) *bad = streamsMismatched.load();
}

double sfz::FilePool::streamClock() const noexcept
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - streamEpoch).count();
}

void sfz::FilePool::enqueueStreamTask() noexcept
{
    threadPool->enqueue([this]() { streamChunkTask(); });
}

double sfz::FilePool::rankOf(const ActiveStream& stream) noexcept
{
    // Every deadline is on the stream clock, seconds since the pool was made: a certain
    // tail's rank, past any of them, keeps the tails in the order they were asked.
    constexpr double kAfterDeadlines { 1e9 };
    return stream.tail ? kAfterDeadlines + stream.startedAt : stream.deadline;
}

void sfz::FilePool::promote(ActiveStream& stream, double startedAt) noexcept
{
    stream.tail = false;
    stream.promoteTo = -1.0;
    stream.startedAt = startedAt;
    stream.deadline = startedAt + (stream.started ? stream.position / stream.sampleRate : 0.0);
}

void sfz::FilePool::removeStream(ActiveStream* stream) noexcept
{
    stream->data->tailPending = false;
    {
        std::lock_guard<std::mutex> lock { streamsMutex };
        const auto it = std::find_if(activeStreams.begin(), activeStreams.end(),
            [stream](const std::unique_ptr<ActiveStream>& p) { return p.get() == stream; });
        if (it != activeStreams.end()) {
            std::iter_swap(it, activeStreams.end() - 1);
            activeStreams.pop_back();
        }
    }
    pendingStreams.fetch_sub(1);
}

sfz::FilePool::StartResult sfz::FilePool::startStream(ActiveStream& stream) noexcept
{
    FileData& data = *stream.data;
    const auto status = data.status.load();
    // Invalid: the preload has not published the file yet. GarbageCollecting: the
    // collector is taking the last stream's buffer. Both pass in moments.
    if (status == FileData::Status::Invalid || status == FileData::Status::GarbageCollecting)
        return ++stream.attempts > 1024 ? StartResult::Drop : StartResult::Retry;
    if (status != FileData::Status::Preloaded)
        return StartResult::Drop;   // already streamed or streaming

    std::shared_ptr<FileId> id = stream.id.lock();
    if (!id)
        return StartResult::Drop;
    std::error_code readError;
    AudioReaderPtr reader = createAudioReader(rootDirectory / id->filename(), id->isReverse(), &readError);
    if (readError || !reader) {
        DBG("[sfizz] reading the file errored for " << *id << " with code " << readError << ": " << readError.message());
        return StartResult::Drop;
    }

    auto expected = FileData::Status::Preloaded;
    if (!data.status.compare_exchange_strong(expected, FileData::Status::Streaming))
        return expected == FileData::Status::GarbageCollecting ? StartResult::Retry : StartResult::Drop;

    const auto numFrames = static_cast<size_t>(reader->frames());
    const unsigned numChannels = reader->channels();

    // The decoder starts where the head ends (a reader that cannot seek, at 0).
    size_t head = std::min(data.headFrames(), numFrames);
    const bool packed = data.packedHead.getNumFrames() > 0;
    const size_t headChannels = packed ? data.packedHead.getNumChannels() : data.preloadedData.getNumChannels();
    if (head == 0 || headChannels != numChannels || !reader->seek(head))
        head = 0;

    // A 16-bit file after a packed head: the frames past it, packed, the head shared. Its
    // decoded floats are k/32768 like the head's, so packing them is exact.
    stream.packed = packed && head > 0 && reader->bitsPerSample() == 16 && packStreams.load();
    if (stream.packed) {
        PackedHeadBuffer& tail = data.packedTail;
        tail.reset();
        tail.addChannels(numChannels);
        tail.resize(numFrames - head);
        tail.clear();
        data.tailStart = head;
        streamsPacked.fetch_add(1);
    }

    // Otherwise the whole file as floats, the head copied in: the same frames this file
    // decodes to (a packed head converts back exactly).
    FileAudioBuffer& out = data.fileData;
    if (stream.packed)
        out.reset();
    else {
        data.packedTail.reset();
        out.reset();
        out.addChannels(numChannels);
        out.resize(numFrames);
        out.clear();
    }
    for (size_t c = 0; c < numChannels && head > 0 && !stream.packed; ++c) {
        const auto to = out.getSpan(c);
        if (packed) {
            const auto from = data.packedHead.getConstSpan(c);
            for (size_t i = 0; i < head; ++i)
                to[i] = static_cast<float>(from[i]) * kPackedHeadScale;
        } else {
            const auto from = data.preloadedData.getConstSpan(c);
            std::copy(from.begin(), from.begin() + head, to.begin());
        }
    }
    data.availableFrames.store(head);   // not past the head: voices keep reading the head
    (head > 0 ? streamsFromHead : streamsFromZero).fetch_add(1);

    stream.reader = std::move(reader);
    stream.numFrames = numFrames;
    stream.position = head;
    stream.tailStart = head;
    stream.sampleRate = stream.reader->sampleRate() > 0 ? stream.reader->sampleRate() : config::defaultSampleRate;
    stream.deadline = stream.startedAt + head / stream.sampleRate;
    stream.started = true;
    return StartResult::Started;
}

bool sfz::FilePool::decodeChunk(ActiveStream& stream) noexcept
{
    FileData& data = *stream.data;
    const unsigned numChannels = stream.reader->channels();
    thread_local sfz::Buffer<float> block;
    const size_t want = std::min(kStreamChunkFrames, stream.numFrames - stream.position);
    if (want == 0)
        return true;
    if (block.size() < want * numChannels)
        block.resize(kStreamChunkFrames * std::max(2u, numChannels));
    const size_t got = stream.reader->readNextBlock(block.data(), want);
    for (unsigned c = 0; c < numChannels && got > 0; ++c) {
        if (stream.packed) {
            const auto to = data.packedTail.getSpan(c).subspan(stream.position - stream.tailStart, got);
            for (size_t i = 0; i < got; ++i)
                to[i] = static_cast<int16_t>(block[i * numChannels + c] * 32768.0f);
        } else {
            const auto to = data.fileData.getSpan(c).subspan(stream.position, got);
            for (size_t i = 0; i < got; ++i)
                to[i] = block[i * numChannels + c];
        }
    }
    stream.position += got;
    stream.decodedFrames += got;
    data.availableFrames.fetch_add(got);
    stream.deadline = stream.startedAt + stream.position / stream.sampleRate;
    return got < want || stream.position >= stream.numFrames;
}

void sfz::FilePool::finishStream(ActiveStream& stream) noexcept
{
    FileData& data = *stream.data;
    if (verifyStreams.load()) {
        bool same = false;
        if (std::shared_ptr<FileId> id = stream.id.lock()) {
            AudioReaderPtr reader = createAudioReader(rootDirectory / id->filename(), id->isReverse());
            const FileAudioBuffer whole = readFromFile(*reader, static_cast<uint32_t>(stream.numFrames));
            const size_t n = data.availableFrames.load();
            const size_t channels = stream.packed ? data.packedTail.getNumChannels() : data.fileData.getNumChannels();
            same = whole.getNumFrames() == n && whole.getNumChannels() == channels;
            for (size_t c = 0; same && c < whole.getNumChannels(); ++c) {
                const auto a = whole.getConstSpan(c);
                if (!stream.packed) {
                    const auto b = data.fileData.getConstSpan(c);
                    same = std::equal(a.begin(), a.begin() + n, b.begin());
                    continue;
                }
                // The head it follows, then the tail, as the voices read them.
                const size_t h = stream.tailStart;
                const auto head = data.packedHead.getConstSpan(c);
                const auto tail = data.packedTail.getConstSpan(c);
                for (size_t i = 0; same && i < n; ++i)
                    same = a[i] == static_cast<float>(i < h ? head[i] : tail[i - h]) * kPackedHeadScale;
            }
        }
        (same ? streamsVerified : streamsMismatched).fetch_add(1);
    }

    data.status = FileData::Status::Done;
    streamBytes.fetch_add(streamedBytes(data), std::memory_order_relaxed);
    streamFiles.fetch_add(1, std::memory_order_relaxed);
    // What the decode speed means now: frames decoded here over the time the tasks spent
    // on them, opening included (the head is copied, and waits between chunks are not
    // decoding).
    recordStreamTiming(stream.decodedFrames, stream.workMicros, stream.firstMicros, stream.waitMicros, stream.ahead);

    if (std::shared_ptr<FileId> id = stream.id.lock()) {
        std::lock_guard<SpinMutex> guard { garbageAndLastUsedMutex };
        if (absl::c_find(lastUsedFiles, *id) == lastUsedFiles.end())
            lastUsedFiles.push_back(*id);
    }
}

void sfz::FilePool::streamChunkTask() noexcept
{
    thread_local bool prioritySet = false;
    if (!prioritySet) {
        raiseCurrentThreadPriority();
        prioritySet = true;
    }

    ActiveStream* stream = nullptr;
    {
        std::lock_guard<std::mutex> lock { streamsMutex };
        for (const auto& p : activeStreams)
            if (!p->busy && (!stream || rankOf(*p) < rankOf(*stream)))
                stream = p.get();
        if (stream)
            stream->busy = true;
    }
    if (!stream) {   // cannot happen with one task per stream; stay in the count
        std::this_thread::yield();
        enqueueStreamTask();
        return;
    }

    bool finished = false;
    bool drop = streamsCancelled.load() || stream->id.expired();
    const auto start = std::chrono::steady_clock::now();
    if (!drop && !stream->started) {
        switch (startStream(*stream)) {
        case StartResult::Started: break;
        case StartResult::Drop: drop = true; break;
        case StartResult::Retry:
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            break;
        }
    }
    const bool first = stream->started && stream->decodedFrames == 0;
    if (!drop && stream->started)
        finished = decodeChunk(*stream);
    const uint64_t work = microsSince(start);
    stream->workMicros += work;
    if (first && !drop) {
        stream->firstMicros = work;
        const double waited = std::chrono::duration<double>(start - streamEpoch).count() - stream->requestedAt;
        stream->waitMicros = waited > 0 ? static_cast<uint64_t>(waited * 1e6) : 0;
    }

    if (finished)
        finishStream(*stream);
    if (finished || drop) {
        if (drop && stream->started)
            stream->data->status = FileData::Status::Done;   // what it has stays readable
        removeStream(stream);
        streamTasks.fetch_sub(1);
        return;
    }
    {
        std::lock_guard<std::mutex> lock { streamsMutex };
        stream->busy = false;
        if (stream->promoteTo >= 0.0)
            promote(*stream, stream->promoteTo);
    }
    enqueueStreamTask();
}

void sfz::FilePool::clear()
{
    std::lock_guard<SpinMutex> guard { garbageAndLastUsedMutex };
    emptyFileLoadingQueues();
    garbageToCollect.clear();
    packedGarbage.clear();
    lastUsedFiles.clear();
    preloadedFiles.clear();
    loadedFiles.clear();
}

uint32_t sfz::FilePool::getPreloadSize() const noexcept
{
    return preloadSize;
}

template <typename R>
bool is_ready(std::future<R> const& f)
{
    return f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

void sfz::FilePool::dispatchingJob() noexcept
{
    while (dispatchBarrier.wait(), dispatchFlag) {
        QueuedFileData queuedData;
        if (!filesToLoad->try_pop(queuedData))
            continue;
        // A deleted region (its file ID nulled), or a file two voices queued in the same
        // block: one stream serves every voice.
        bool skip = queuedData.id.expired();
        auto stream = std::unique_ptr<ActiveStream>(new ActiveStream);
        stream->id = queuedData.id;
        stream->data = queuedData.data;
        stream->requestedAt = queuedData.queuedAt;
        stream->startedAt = queuedData.startedAt;
        stream->deadline = stream->startedAt;
        stream->tail = queuedData.tail;
        {
            std::lock_guard<std::mutex> lock { streamsMutex };
            for (const auto& p : activeStreams) {
                if (p->data != queuedData.data)
                    continue;
                skip = true;
                // A tail started (9d): its stream takes the tail's deadline, now or, while a
                // task decodes it, when that task gives it back.
                if (p->tail && !queuedData.tail) {
                    tailsMovedUp.fetch_add(1, std::memory_order_relaxed);
                    if (p->busy)
                        p->promoteTo = queuedData.startedAt;
                    else
                        promote(*p, queuedData.startedAt);
                }
            }
            stream->ahead = static_cast<int>(activeStreams.size());
            if (!skip)
                activeStreams.push_back(std::move(stream));
        }
        if (skip) {
            pendingStreams.fetch_sub(1);
            continue;
        }
        streamTasks.fetch_add(1);
        enqueueStreamTask();
    }
}

void sfz::FilePool::garbageJob() noexcept
{
    while (semGarbageBarrier.wait(), garbageFlag) {
        std::lock_guard<SpinMutex> guard { garbageAndLastUsedMutex };
        garbageToCollect.clear();
        packedGarbage.clear();
    }
}

void sfz::FilePool::waitForBackgroundLoading() noexcept
{
    // Every stream queued so far, to its end (freewheeling renders, clearing the pool).
    while (pendingStreams.load() > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
}

void sfz::FilePool::raiseCurrentThreadPriority() noexcept
{
#if defined(_WIN32)
    HANDLE thread = GetCurrentThread();
    const int priority = THREAD_PRIORITY_ABOVE_NORMAL; /*THREAD_PRIORITY_HIGHEST*/
    if (!SetThreadPriority(thread, priority)) {
        std::system_error error(GetLastError(), std::system_category());
        DBG("[sfizz] Cannot set current thread priority: " << error.what());
    }
#else
    pthread_t thread = pthread_self();
    int policy;
    sched_param param;

    if (pthread_getschedparam(thread, &policy, &param) != 0) {
        DBG("[sfizz] Cannot get current thread scheduling parameters");
        return;
    }

    policy = SCHED_RR;
    const int minprio = sched_get_priority_min(policy);
    const int maxprio = sched_get_priority_max(policy);
    param.sched_priority = minprio + config::backgroundLoaderPthreadPriority * (maxprio - minprio) / 100;

    if (pthread_setschedparam(thread, policy, &param) != 0) {
        DBG("[sfizz] Cannot set current thread scheduling parameters");
        return;
    }
#endif
}

void sfz::FilePool::setRamLoading(bool loadInRam) noexcept
{
    if (loadInRam == this->loadInRam)
        return;

    this->loadInRam = loadInRam;

    if (loadInRam) {
        for (auto& preloadedFile : preloadedFiles) {
            fs::path file { rootDirectory / preloadedFile.first.filename() };
            AudioReaderPtr reader = createAudioReader(file, preloadedFile.first.isReverse());
            auto& fileData = preloadedFile.second;
            fileData.setHead(readFromFile(
                *reader,
                fileData.information.end
            ));
            fileData.fullyLoaded = true;
        }
    } else {
        setPreloadSize(preloadSize);
    }
}

// The budget's pass (design 4f). Called with garbageAndLastUsedMutex held, on the audio
// thread: no allocation (gcCandidates is reserved like lastUsedFiles). Files a voice reads
// are kept whatever the budget (the active streams, a separate term in the app's sum).
void sfz::FilePool::collectOverBudget(int64_t budget, int& keptReading, int& keptRecent, int& freed) noexcept
{
    gcCandidates.clear();
    for (const FileId& id : lastUsedFiles) {
        auto it = preloadedFiles.find(id);
        if (it == preloadedFiles.end())
            continue;
        FileData& data = it->second;
        if (data.readerCount != 0) {
            ++keptReading;
            continue;
        }
        if (data.status.load() == FileData::Status::Done && gcCandidates.size() < gcCandidates.capacity())
            gcCandidates.push_back(&data);
    }

    int64_t over = streamBytes.load(std::memory_order_relaxed) - budget;
    if (over > 0) {
        // least recently idle first
        std::sort(gcCandidates.begin(), gcCandidates.end(), [](const FileData* a, const FileData* b) {
            return a->lastViewerLeftAt < b->lastViewerLeftAt;
        });
        for (FileData* data : gcCandidates) {
            if (over <= 0 || garbageToCollect.size() == garbageToCollect.capacity())
                break;
            auto status = FileData::Status::Done;
            if (!data->status.compare_exchange_strong(status, FileData::Status::GarbageCollecting))
                continue;
            if (data->readerCount.load() != 0) {
                data->status = FileData::Status::Done;
                continue;
            }
            const int64_t bytes = streamedBytes(*data);
            data->availableFrames = 0;
            streamBytes.fetch_sub(bytes, std::memory_order_relaxed);
            streamFiles.fetch_sub(1, std::memory_order_relaxed);
            over -= bytes;
            ++freed;
            garbageToCollect.push_back(std::move(data->fileData));
            packedGarbage.push_back(std::move(data->packedTail));
            data->packedTail.reset();   // a moved-from buffer keeps its sizes
            data->status = FileData::Status::Preloaded;
        }
    }
    keptRecent = static_cast<int>(gcCandidates.size()) - freed;

    // Off the list: what was freed (now or before) and what the pool no longer has.
    swapAndPopAll(lastUsedFiles, [&](const FileId& id) {
        auto it = preloadedFiles.find(id);
        if (it == preloadedFiles.end())
            return true;
        const FileData& data = it->second;
        return data.status.load() == FileData::Status::Preloaded && data.availableFrames == 0;
    });
}

void sfz::FilePool::triggerGarbageCollection() noexcept
{
    gcDue.fetch_add(1, std::memory_order_relaxed);
    const std::unique_lock<SpinMutex> guard { garbageAndLastUsedMutex, std::try_to_lock };
    if (!guard.owns_lock()) {
        gcLockMissed.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    const auto now = std::chrono::high_resolution_clock::now();
    int keptReading = 0, keptRecent = 0, freed = 0;
    const int64_t budget = cacheBudgetBytes.load(std::memory_order_relaxed);
    if (budget >= 0) {
        collectOverBudget(budget, keptReading, keptRecent, freed);
    } else swapAndPopAll(lastUsedFiles, [&](const FileId& id) {
        if (garbageToCollect.size() == garbageToCollect.capacity())
           return false;

        auto it = preloadedFiles.find(id);
        if (it == preloadedFiles.end()) {
            // Getting here means that the preloadedFiles got changed (probably cleared)
            // while the lastUsedFiles were untouched.
            return true;
        }

        sfz::FileData& data = it->second;

        if (data.readerCount != 0) {
            ++keptReading;
            return false;
        }

        const auto secondsIdle = std::chrono::duration_cast<std::chrono::seconds>(now - data.lastViewerLeftAt).count();
        if (secondsIdle < config::fileClearingPeriod) {
            ++keptRecent;
            return false;
        }

        auto status = data.status.load();
        if (status == FileData::Status::Preloaded) {
            // do the garbage collection when availableFrames != 0
            if (data.availableFrames == 0) {
                return true;
            }
        }
        else if (status != FileData::Status::Done) {
            return false;
        }

        // do garbage collection when changing the status is success
        if (data.status.compare_exchange_strong(status, FileData::Status::GarbageCollecting)) {
            // recheck readerCount
            auto readerCount = data.readerCount.load();
            if (readerCount == 0) {
                data.availableFrames = 0;
                streamBytes.fetch_sub(streamedBytes(data), std::memory_order_relaxed);
                streamFiles.fetch_sub(1, std::memory_order_relaxed);
                ++freed;
                garbageToCollect.push_back(std::move(data.fileData));
                packedGarbage.push_back(std::move(data.packedTail));
                data.packedTail.reset();
                data.status = FileData::Status::Preloaded;
                return true;
            }
            data.status = status;
        }
        return false;
    });
    gcPasses.fetch_add(1, std::memory_order_relaxed);
    gcKeptReading.store(keptReading, std::memory_order_relaxed);
    gcKeptRecent.store(keptRecent, std::memory_order_relaxed);
    gcFreed.store(freed, std::memory_order_relaxed);
    gcPending.store(static_cast<int>(lastUsedFiles.size()), std::memory_order_relaxed);

    std::error_code ec;
    semGarbageBarrier.post(ec);
    ASSERT(!ec);
}
