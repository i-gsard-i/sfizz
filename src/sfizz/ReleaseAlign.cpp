// SPDX-License-Identifier: BSD-2-Clause
//
// See ReleaseAlign.h. Global registry for phase-aligned release tails.

#include "ReleaseAlign.h"
#include <atomic>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace {

struct Entry { float period; int anchor; float loopRms; std::vector<uint8_t> track; };

struct Registry {
    std::unordered_map<std::string, Entry> map;
    std::atomic<bool> enabled { false };
    std::atomic<float> crossfade { 0.025f };
    std::atomic<float> fallback { 1.4f };   // no-tail fade (~typical recorded-tail length)
};

// Function-local static avoids static-init-order issues; constructed on first use,
// which happens at load (engine thread) well before audio starts.
Registry& reg()
{
    static Registry r;
    return r;
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

} // namespace sfz

extern "C" {

void sfizz_release_align_set(const char* attackSample, float period, int anchor, float loopRms)
{
    if (!attackSample || period <= 0.0f)
        return;
    reg().map[sfz::releaseAlignKey(attackSample)] = Entry { period, anchor, loopRms, {} };
}

int sfizz_release_align_set_track(const char* attackSample, const char* column)
{
    if (!attackSample || !column || std::strncmp(column, "v1:", 3) != 0)
        return 0;
    auto it = reg().map.find(sfz::releaseAlignKey(attackSample));
    if (it == reg().map.end())
        return 0;
    // base64, stopping at the first character outside the alphabet ('=', tab, CR, end).
    std::vector<uint8_t> out;
    uint32_t acc = 0;
    int bits = 0;
    for (const char* c = column + 3; *c; ++c) {
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
    it->second.track = std::move(out);
    return static_cast<int>(it->second.track.size());
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

} // extern "C"
