// SPDX-License-Identifier: BSD-2-Clause

// This code is part of the sfizz library and is licensed under a BSD 2-clause
// license. You should have receive a LICENSE.md file along with the code.
// If not, contact the sfizz maintainers at https://github.com/sfztools/sfizz

#pragma once
#include "absl/types/span.h"
#include "ghc/fs_std.hpp"
#include <st_audiofile.h>
#include <system_error>
#include <memory>
#include <cstdio>

namespace sfz {
struct InstrumentInfo;
struct WavetableInfo;

/**
 * @brief Designation of a particular kind of audio reader
 */
enum class AudioReaderType {
    //! Reader in forward direction
    Forward,
    //! Reader in reverse direction
    Reverse,
    //! Reader in reverse direction, operating on a whole file instead of seeking
    NoSeekReverse,
};

/**
 * @brief Reader of audio file data
 */
class AudioReader {
public:
    virtual ~AudioReader() {}
    virtual AudioReaderType type() const = 0;
    virtual int format() const = 0;
    virtual int64_t frames() const = 0;
    virtual unsigned channels() const = 0;
    virtual unsigned sampleRate() const = 0;
    // organdroid: bits of an integer PCM file, 0 when unknown (a stream of a 16-bit file
    // is kept as 16-bit, FilePool.cpp).
    virtual unsigned bitsPerSample() const { return 0; }
    virtual size_t readNextBlock(float* buffer, size_t frames) = 0;
    // organdroid: move to a frame, so a stream can start after the preloaded head.
    // Forward readers only; false means the reader cannot, and reads from where it is.
    virtual bool seek(uint64_t frame) { (void)frame; return false; }
    virtual bool getInstrumentInfo(InstrumentInfo&) { return false; };
    virtual bool getWavetableInfo(WavetableInfo&) { return false; };
};

typedef std::unique_ptr<AudioReader> AudioReaderPtr;

/**
 * @brief Create a file reader of detected type.
 */
AudioReaderPtr createAudioReader(const fs::path& path, bool reverse, std::error_code* ec = nullptr);

/**
 * @brief Create a memory reader of detected type.
 */
AudioReaderPtr createAudioReaderFromMemory(const void* memory, size_t length, bool reverse, std::error_code* ec = nullptr);

} // namespace sfz
