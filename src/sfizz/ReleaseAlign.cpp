// SPDX-License-Identifier: BSD-2-Clause
//
// See ReleaseAlign.h. Global registry for phase-aligned release tails.

#include "ReleaseAlign.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

struct Entry {
    float period; int anchor; float loopRms;
    std::vector<uint8_t> track; std::vector<uint8_t> onset; std::vector<uint8_t> tail;
};

struct Registry {
    std::unordered_map<std::string, Entry> map;
    std::atomic<bool> enabled { false };
    std::atomic<float> crossfade { 0.025f };
    std::atomic<float> fallback { 1.4f };   // no-tail fade (~typical recorded-tail length)
    std::atomic<float> tailCut { 0.0f };    // 0 = recorded tails play whole
    std::atomic<int> restrike { 0 };        // 0 by organ, 1 always, 2 never
};

// Function-local static avoids static-init-order issues; constructed on first use,
// which happens at load (engine thread) well before audio starts.
Registry& reg()
{
    static Registry r;
    return r;
}

// base64, stopping at the first character outside the alphabet ('=', tab, CR, end).
std::vector<uint8_t> decodeBase64(const char* c)
{
    std::vector<uint8_t> out;
    uint32_t acc = 0;
    int bits = 0;
    for (; *c; ++c) {
        int v;
        if (*c >= 'A' && *c <= 'Z') v = *c - 'A';
        else if (*c >= 'a' && *c <= 'z') v = *c - 'a' + 26;
        else if (*c >= '0' && *c <= '9') v = *c - '0' + 52;
        else if (*c == '+') v = 62;
        else if (*c == '/') v = 63;
        else break;
        acc = (acc << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

} // namespace

namespace sfz {

std::string releaseAlignKey(const std::string& path)
{
    // Exact path match, normalized to '/' separators only. Both sides of the lookup
    // already agree on the path form within a single load: the legacy master SFZ uses
    // relative sample paths ("Subbas16/036-C.flac") and ships a tsv keyed the same way,
    // while the multi-organ palette rewrites sample= to absolute per-organ paths
    // (".../files/<organId>/samples/Subbas16/036-C.flac") and the app writes the tsv with
    // those same absolute keys. Keeping the WHOLE path (not just the last two components)
    // is what makes the key organ-unique: a relative "Principal8/036-C.flac" would collide
    // across organs in a mixed palette, but the absolute path carries the organ directory.
    std::string out = path;
    for (char& c : out)
        if (c == '\\')
            c = '/';
    return out;
}

bool releaseAlignEnabled() noexcept
{
    return reg().enabled.load(std::memory_order_relaxed);
}

float releaseAlignCrossfade() noexcept
{
    return reg().crossfade.load(std::memory_order_relaxed);
}

float releaseAlignFallback() noexcept
{
    return reg().fallback.load(std::memory_order_relaxed);
}

float releaseTailCut() noexcept
{
    return reg().tailCut.load(std::memory_order_relaxed);
}

bool restrikeEndsTail(bool wet) noexcept
{
    const int mode = reg().restrike.load(std::memory_order_relaxed);
    return mode == 1 || (mode == 0 && !wet);
}

bool releaseAlignLookup(const std::string& attackSample, float& period, int& anchor, float& loopRms,
                        const uint8_t*& track, size_t& trackSize)
{
    const auto& m = reg().map;
    auto it = m.find(releaseAlignKey(attackSample));
    if (it == m.end())
        return false;
    period = it->second.period;
    anchor = it->second.anchor;
    loopRms = it->second.loopRms;
    track = it->second.track.empty() ? nullptr : it->second.track.data();
    trackSize = it->second.track.size();
    return true;
}

float releaseAlignTrackCorrection(const uint8_t* track, size_t trackSize, float period, int64_t pos) noexcept
{
    if (!track || trackSize == 0 || period <= 0.0f)
        return 0.0f;
    const int64_t spacing = 4 * static_cast<int64_t>(std::lround(period));
    if (spacing <= 0)
        return 0.0f;
    int64_t i = pos > 0 ? (pos + spacing / 2) / spacing : 0;    // nearest point
    if (i >= static_cast<int64_t>(trackSize))
        i = static_cast<int64_t>(trackSize) - 1;
    return track[i] * period / 256.0f;
}

bool releaseAlignOnset(const std::string& attackSample, const uint8_t*& onset, size_t& onsetSize)
{
    const auto& m = reg().map;
    auto it = m.find(releaseAlignKey(attackSample));
    if (it == m.end() || it->second.onset.empty())
        return false;
    onset = it->second.onset.data();
    onsetSize = it->second.onset.size();
    return true;
}

int64_t restrikeEntry(const uint8_t* onset, size_t onsetSize, float period, float level) noexcept
{
    const int want = static_cast<int>(std::ceil(255.0f * std::min(std::max(level, 0.0f), 1.0f)));
    size_t k = 0;
    while (k < onsetSize && onset[k] < want)
        ++k;
    return static_cast<int64_t>(k) * static_cast<int64_t>(std::lround(period));
}

bool releaseAlignTail(const std::string& attackSample, const uint8_t*& tail, size_t& tailSize)
{
    const auto& m = reg().map;
    auto it = m.find(releaseAlignKey(attackSample));
    if (it == m.end() || it->second.tail.size() < 3)
        return false;
    tail = it->second.tail.data();
    tailSize = it->second.tail.size();
    return true;
}

int64_t restrikeTailEntry(const uint8_t* tail, size_t tailSize, float period, int64_t pos) noexcept
{
    const int64_t spacing = std::lround(period);
    if (!tail || tailSize < 3 || spacing <= 0)
        return -1;
    const int64_t points = 1 + static_cast<int64_t>(tailSize - 3) / 2;
    const int64_t i = pos > 0 ? (pos + spacing / 2) / spacing : 0;    // nearest point
    if (i >= points)
        return -1;   // the tail is quieter than the onset resolves
    // The steps are summed up to the point: a few thousand at most, once per re-strike.
    int64_t entry = tail[0] | (tail[1] << 8) | (static_cast<int64_t>(tail[2]) << 16);
    for (int64_t j = 0; j < i; ++j) {
        const uint8_t* b = tail + 3 + 2 * j;
        entry += static_cast<int16_t>(b[0] | (b[1] << 8)) + spacing;
    }
    return std::max<int64_t>(entry + pos - i * spacing, 0);
}

} // namespace sfz

extern "C" {

void sfizz_release_align_set(const char* attackSample, float period, int anchor, float loopRms)
{
    if (!attackSample || period <= 0.0f)
        return;
    reg().map[sfz::releaseAlignKey(attackSample)] = Entry { period, anchor, loopRms, {}, {}, {} };
}

int sfizz_release_align_set_track(const char* attackSample, const char* column)
{
    if (!attackSample || !column || std::strncmp(column, "v1:", 3) != 0)
        return 0;
    auto it = reg().map.find(sfz::releaseAlignKey(attackSample));
    if (it == reg().map.end())
        return 0;
    it->second.track = decodeBase64(column + 3);
    return static_cast<int>(it->second.track.size());
}

int sfizz_release_align_set_onset(const char* attackSample, const char* column)
{
    if (!attackSample || !column || std::strncmp(column, "o1:", 3) != 0)
        return 0;
    auto it = reg().map.find(sfz::releaseAlignKey(attackSample));
    if (it == reg().map.end())
        return 0;
    it->second.onset = decodeBase64(column + 3);
    return static_cast<int>(it->second.onset.size());
}

int sfizz_release_align_set_tail(const char* attackSample, const char* column)
{
    if (!attackSample || !column || std::strncmp(column, "t2:", 3) != 0)
        return 0;
    auto it = reg().map.find(sfz::releaseAlignKey(attackSample));
    if (it == reg().map.end())
        return 0;
    it->second.tail = decodeBase64(column + 3);
    const size_t n = it->second.tail.size();
    return n < 3 ? 0 : static_cast<int>(1 + (n - 3) / 2);
}

void sfizz_release_align_clear()
{
    reg().map.clear();
}

void sfizz_release_align_enable(int on)
{
    reg().enabled.store(on != 0, std::memory_order_relaxed);
}

void sfizz_release_align_set_crossfade(float seconds)
{
    if (seconds > 0.0f)
        reg().crossfade.store(seconds, std::memory_order_relaxed);
}

void sfizz_release_align_set_fallback(float seconds)
{
    if (seconds > 0.0f)
        reg().fallback.store(seconds, std::memory_order_relaxed);
}

void sfizz_release_tail_cut(float seconds)
{
    reg().tailCut.store(seconds > 0.0f ? seconds : 0.0f, std::memory_order_relaxed);
}

void sfizz_restrike_ends_tails(int mode)
{
    reg().restrike.store(mode >= 0 && mode <= 2 ? mode : 0, std::memory_order_relaxed);
}

} // extern "C"
