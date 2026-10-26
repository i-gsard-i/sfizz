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
// stream (whole-file frames, the time from opening the file to the last frame). Worker
// threads write, the app reads; nothing here runs on the audio thread. Stream timings are
// kept in a small buffer the app drains; when it is full the oldest go.

namespace {
struct DecodeTiming { uint32_t frames; uint32_t micros; };
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

void recordStreamTiming(uint64_t frames, uint64_t micros)
{
    std::lock_guard<std::mutex> lock { streamTimingsMutex };
    if (streamTimings.size() >= kMaxStreamTimings)
        streamTimings.erase(streamTimings.begin());
    streamTimings.push_back({ static_cast<uint32_t>(frames), static_cast<uint32_t>(std::min<uint64_t>(micros, UINT32_MAX)) });
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
extern "C" int sfizz_stream_timings(uint32_t* frames, uint32_t* micros, int max)
{
    std::lock_guard<std::mutex> lock { streamTimingsMutex };
    const int n = std::min<int>(max, static_cast<int>(streamTimings.size()));
    for (int i = 0; i < n; ++i) {
        frames[i] = streamTimings[i].frames;
        micros[i] = streamTimings[i].micros;
    }
    streamTimings.erase(streamTimings.begin(), streamTimings.begin() + n);
    return n;
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

void streamFromFile(sfz::AudioReader& reader, sfz::FileAudioBuffer& output, std::atomic<size_t>* filledFrames = nullptr)
{
    const auto numFrames = static_cast<size_t>(reader.frames());
    const auto numChannels = reader.channels();
    const auto chunkSize = static_cast<size_t>(sfz::config::fileChunkSize);

    output.reset();
    output.addChannels(reader.channels());
    output.resize(numFrames);
    output.clear();

    sfz::Buffer<float> fileBlock { chunkSize * numChannels };
    size_t inputFrameCounter { 0 };
    size_t outputFrameCounter { 0 };
    bool inputEof = false;

    while (!inputEof && inputFrameCounter < numFrames)
    {
        auto thisChunkSize = std::min(chunkSize, numFrames - inputFrameCounter);
        const auto numFramesRead = static_cast<size_t>(
            reader.readNextBlock(fileBlock.data(), thisChunkSize));
        if (numFramesRead == 0)
            break;

        if (numFramesRead < thisChunkSize) {
            inputEof = true;
            thisChunkSize = numFramesRead;
        }
        const auto outputChunkSize = thisChunkSize;

        for (size_t chanIdx = 0; chanIdx < numChannels; chanIdx++) {
            const auto outputChunk = output.getSpan(chanIdx).subspan(outputFrameCounter, outputChunkSize);
            for (size_t i = 0; i < thisChunkSize; ++i)
                outputChunk[i] = fileBlock[i * numChannels + chanIdx];
        }
        inputFrameCounter += thisChunkSize;
        outputFrameCounter += outputChunkSize;

        if (filledFrames != nullptr)
            filledFrames->fetch_add(outputChunkSize);
    }
}

sfz::FilePool::FilePool()
    : filesToLoad(alignedNew<FileQueue>()),
      threadPool(globalThreadPool())
{
    loadingJobs.reserve(config::maxVoices);
    lastUsedFiles.reserve(config::maxVoices);
    garbageToCollect.reserve(config::maxVoices);
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

    for (auto& job : loadingJobs)
        job.wait();
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

bool sfz::FilePool::preloadFile(const FileId& fileId, uint32_t maxOffset) noexcept
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
    const fs::path file { rootDirectory / fileId.filename() };
    AudioReaderPtr reader = createAudioReader(file, fileId.isReverse());

    const auto frames = static_cast<uint32_t>(reader->frames());
    const auto framesToLoad = [&]() {
        if (loadInRam)
            return frames;
        else
            return min(frames, maxOffset + preloadSize);
    }();

    const auto existingFile = preloadedFiles.find(fileId);
    if (existingFile != preloadedFiles.end()) {
        auto& fileData = existingFile->second;
        if (framesToLoad > fileData.headFrames()) {
            fileData.information.maxOffset = maxOffset;
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

void sfz::FilePool::preloadFiles(const std::vector<std::pair<FileId, uint32_t>>& files) noexcept
{
    preloadProgressDone.store(0, std::memory_order_relaxed);
    preloadProgressTotal.store(static_cast<int>(files.size()), std::memory_order_relaxed);

    // Files already tracked keep preloadFile()'s exact sequential bookkeeping; only
    // files in neither map (the whole set on a cold load) are farmed out to workers.
    struct PreloadJob {
        const FileId* id;
        uint32_t maxOffset;
        absl::optional<FileInformation> information;
        FileData data;          // the head set in the worker (setHead packs it there)
        bool fullyLoaded { false };
        bool ok { false };
    };
    std::vector<PreloadJob> jobs;
    jobs.reserve(files.size());
    for (const auto& item : files) {
        if (loadedFiles.contains(item.first) || preloadedFiles.contains(item.first)) {
            preloadFile(item.first, item.second);
            preloadProgressDone.fetch_add(1, std::memory_order_relaxed);
        } else {
            jobs.emplace_back();
            jobs.back().id = &item.first;
            jobs.back().maxOffset = item.second;
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
                        information->sampleRate = static_cast<double>(reader->sampleRate());
                        const auto frames = static_cast<uint32_t>(reader->frames());
                        const auto framesToLoad = loadInRam
                            ? frames : min(frames, job.maxOffset + preloadSize);
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

sfz::FileDataHolder sfz::FilePool::getFilePromise(const std::shared_ptr<FileId>& fileId) noexcept
{
    const auto loaded = loadedFiles.find(*fileId);
    if (loaded != loadedFiles.end())
        return { &loaded->second };

    const auto preloaded = preloadedFiles.find(*fileId);
    if (preloaded == preloadedFiles.end()) {
        DBG("[sfizz] File not found in the preloaded files: " << fileId->filename());
        return {};
    }

    auto& fileData = preloaded->second;
    // organdroid: queue the stream only when nobody has yet. An organ chord strikes the
    // same files over and over (every stop x every key, then their release tails), and a
    // job for a file already Streaming/Done returns at once (loadingJob), so those pushes
    // only filled the 256-slot queue; a full queue then refused the voice outright.
    const auto status = fileData.status.load();
    const bool needsStream = status == FileData::Status::Preloaded
                          || status == FileData::Status::GarbageCollecting;
    if (!fileData.fullyLoaded && needsStream) {
        QueuedFileData queuedData { fileId, &fileData };
        if (!filesToLoad->try_push(queuedData)) {
            // Still full: play from the preloaded head rather than dropping the note. The
            // voice ends early only if it outruns the head before a later voice queues
            // the stream.
            DBG("[sfizz] Could not enqueue the file to load for " << fileId << " (queue capacity " << filesToLoad->capacity() << ")");
            return { &preloaded->second };
        }

        std::error_code ec;
        dispatchBarrier.post(ec);
        ASSERT(!ec);
    }

    return { &preloaded->second };
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
        const auto maxOffset = fileData.information.maxOffset;
        fs::path file { rootDirectory / fileId.filename() };
        AudioReaderPtr reader = createAudioReader(file, fileId.isReverse());
        const auto frames = reader->frames();
        const auto framesToLoad = min(frames, maxOffset + preloadSize);
        fileData.setHead(readFromFile(*reader, static_cast<uint32_t>(framesToLoad)));
        fileData.fullyLoaded = frames == framesToLoad;
    }
}

void sfz::FilePool::loadingJob(const QueuedFileData& data) noexcept
{
    raiseCurrentThreadPriority();

    std::shared_ptr<FileId> id = data.id.lock();
    if (!id) {
        // file ID was nulled, it means the region was deleted, ignore
        return;
    }

    const auto jobStart = std::chrono::steady_clock::now();
    const fs::path file { rootDirectory / id->filename() };
    std::error_code readError;
    AudioReaderPtr reader = createAudioReader(file, id->isReverse(), &readError);

    if (readError) {
        DBG("[sfizz] reading the file errored for " << *id << " with code " << readError << ": " << readError.message());
        return;
    }

    FileData::Status currentStatus;

    unsigned spinCounter { 0 };

    while (1) {
        currentStatus = data.data->status.load();
        while (currentStatus == FileData::Status::Invalid) {
            // Spin until the state changes
            if (spinCounter > 1024) {
                DBG("[sfizz] " << *id << " is stuck on Invalid? Leaving the load");
                return;
            }

            std::this_thread::sleep_for(std::chrono::microseconds(100));
            currentStatus = data.data->status.load();
            spinCounter += 1;
        }
        // wait for garbage collection
        if (currentStatus == FileData::Status::GarbageCollecting) {
            atomic_queue::spin_loop_pause();
            atomic_queue::spin_loop_pause();
            atomic_queue::spin_loop_pause();
            atomic_queue::spin_loop_pause();
            continue;
        }
        // Already loading or loaded
        if (currentStatus != FileData::Status::Preloaded)
            return;

        // go outside loop if this gets token
        if (data.data->status.compare_exchange_strong(currentStatus, FileData::Status::Streaming))
            break;
    }

    streamFromFile(*reader, data.data->fileData, &data.data->availableFrames);

    data.data->status = FileData::Status::Done;
    // Opening included: a stream pays it too. The wait for the Preloaded state is
    // normally nothing and is counted with it.
    recordStreamTiming(data.data->availableFrames.load(), microsSince(jobStart));

    std::lock_guard<SpinMutex> guard { garbageAndLastUsedMutex };
    if (absl::c_find(lastUsedFiles, *id) == lastUsedFiles.end())
        lastUsedFiles.push_back(*id);
}

void sfz::FilePool::clear()
{
    std::lock_guard<SpinMutex> guard { garbageAndLastUsedMutex };
    emptyFileLoadingQueues();
    garbageToCollect.clear();
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
        std::lock_guard<std::mutex> guard { loadingJobsMutex };

        QueuedFileData queuedData;
        if (filesToLoad->try_pop(queuedData)) {
            if (queuedData.id.expired()) {
                // file ID was nulled, it means the region was deleted, ignore
            }
            else
                loadingJobs.push_back(
                    threadPool->enqueue([this](const QueuedFileData& data) { loadingJob(data); }, std::move(queuedData)));
        }

        // Clear finished jobs
        swapAndPopAll(loadingJobs, [](std::future<void>& future) {
            return is_ready(future);
        });
    }
}

void sfz::FilePool::garbageJob() noexcept
{
    while (semGarbageBarrier.wait(), garbageFlag) {
        std::lock_guard<SpinMutex> guard { garbageAndLastUsedMutex };
        garbageToCollect.clear();
    }
}

void sfz::FilePool::waitForBackgroundLoading() noexcept
{
    std::lock_guard<std::mutex> guard { loadingJobsMutex };

    for (auto& job : loadingJobs)
        job.wait();

    loadingJobs.clear();
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

void sfz::FilePool::triggerGarbageCollection() noexcept
{
    const std::unique_lock<SpinMutex> guard { garbageAndLastUsedMutex, std::try_to_lock };
    if (!guard.owns_lock())
        return;

    const auto now = std::chrono::high_resolution_clock::now();
    swapAndPopAll(lastUsedFiles, [&](const FileId& id) {
        if (garbageToCollect.size() == garbageToCollect.capacity())
           return false;

        auto it = preloadedFiles.find(id);
        if (it == preloadedFiles.end()) {
            // Getting here means that the preloadedFiles got changed (probably cleared)
            // while the lastUsedFiles were untouched.
            return true;
        }

        sfz::FileData& data = it->second;

        if (data.readerCount != 0)
            return false;

        const auto secondsIdle = std::chrono::duration_cast<std::chrono::seconds>(now - data.lastViewerLeftAt).count();
        if (secondsIdle < config::fileClearingPeriod)
            return false;

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
                garbageToCollect.push_back(std::move(data.fileData));
                data.status = FileData::Status::Preloaded;
                return true;
            }
            data.status = status;
        }
        return false;
    });

    std::error_code ec;
    semGarbageBarrier.post(ec);
    ASSERT(!ec);
}
