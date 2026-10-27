// SPDX-License-Identifier: BSD-2-Clause
// See ReleasePacing.h.

#include "ReleasePacing.h"
#include <algorithm>
#include <atomic>
#include <chrono>

namespace {
// Set from the UI thread, read on the render thread at a release
std::atomic<int>   g_cap { 10 };
std::atomic<float> g_target { 0.8f };
std::atomic<float> g_startCost { 1.28f };
// The render thread's callbacks (DivisionGate.cpp's discipline: per thread)
thread_local sfz::ReleasePacingTick g_tick { 0, 0, 0.0f };
std::atomic<int> g_paced { 0 };
std::atomic<int> g_largest { 0 };
std::atomic<int> g_last { 0 };
std::atomic<float> g_measured { 0.0f };   // the last estimate, for the report
// What became of the strikes, for the report: measured; no quiet callback before; ended by
// other starts; no rise after (counted in all the same)
enum { kMeasured, kNoBefore, kOther, kNoRise, kOutcomes };
std::atomic<int> g_strikes[kOutcomes] {};
void count(int outcome) { g_strikes[outcome].fetch_add(1, std::memory_order_relaxed); }

// The start cost measured live, per thread, from the chords struck (not from releases).
// Each callback is timed from its report to the end of its last render. A strike is a run of
// callbacks starting struck pipes, between a callback before it and one after it that start
// nothing. Whatever stays the same across them (the callback's fixed part, the pipes already
// held, the reverb) cancels in the differences:
//   start cost = (strike - before) / (after - before)
// the strike's callbacks summed, less the pipes they started earlier and now only sustain
// (a chord struck by hand spreads over a few callbacks). Summed over all strikes, so a big
// chord weighs more than a single pipe, and a stray preempted callback is diluted. (Not left
// out past the callback's deadline: on the M31 a full chord's strike takes longer than its
// callback, which the buffer absorbs; that is the cost to measure.)
using Clock = std::chrono::steady_clock;
struct Measure {
    Clock::time_point start {};
    Clock::time_point renderEnd {};
    bool rendered = false;
    int struck = 0;         // pipes started by note-ons in this callback
    int other = 0;          // other starts: release tails, composites handing over
    float before = -1.0f;   // the last callback that started nothing; < 0: none yet
    // the strike in progress
    int strikeCallbacks = 0;
    float strikeExcess = 0.0f;    // the strike's callbacks less `before`, summed
    long strikeEarlier = 0;       // the pipes each callback carries from earlier in the strike
    int strikePipes = 0;
    double numerator = 0.0;       // over all strikes
    double denominator = 0.0;
};
thread_local Measure g_measure;

void finishCallback(Measure& m)
{
    if (!m.rendered)
        return;
    const float t = std::chrono::duration<float>(m.renderEnd - m.start).count();
    if (m.other == 0 && m.struck > 0 && m.before < 0.0f && m.strikeCallbacks == 0)
        count(kNoBefore);
    if (m.other == 0 && m.struck > 0 && m.before >= 0.0f) {   // a strike begins or goes on
        m.strikeExcess += t - m.before;
        m.strikeEarlier += m.strikePipes;
        m.strikePipes += m.struck;
        ++m.strikeCallbacks;
        return;
    }
    if (m.strikeCallbacks > 0 && m.other == 0 && m.struck == 0) {
        // The struck pipes, sounding. Kept when the noise makes it <= 0: leaving those out
        // would bias the sums up (a few pipes struck rise little above the noise).
        const float sustain = t - m.before;
        m.numerator += m.strikeExcess - sustain * m.strikeEarlier / m.strikePipes;
        m.denominator += sustain;
        if (m.denominator > 0.0)
            g_measured.store(static_cast<float>(m.numerator / m.denominator), std::memory_order_relaxed);
        count(sustain > 0.0f ? kMeasured : kNoRise);
    } else if (m.strikeCallbacks > 0) {
        count(kOther);
    }
    m.strikeCallbacks = 0;
    m.strikeExcess = 0.0f;
    m.strikeEarlier = 0;
    m.strikePipes = 0;
    m.before = (m.struck == 0 && m.other == 0) ? t : -1.0f;
}
}

extern "C" void sfizz_release_pacing(int cap, float target, float startCost)
{
    g_cap.store(cap, std::memory_order_relaxed);
    g_target.store(target, std::memory_order_relaxed);
    g_startCost.store(startCost, std::memory_order_relaxed);
}

extern "C" void sfizz_release_pacing_callback(int frames, float load)
{
    g_tick.serial = g_tick.serial + 1 == 0 ? 1 : g_tick.serial + 1;
    g_tick.frames = frames;
    g_tick.load = load;
    Measure& m = g_measure;
    finishCallback(m);
    m.start = Clock::now();
    m.rendered = false;
    m.struck = 0;
    m.other = 0;
}

extern "C" void sfizz_release_pacing_counts(int* paced, int* largest, int* last)
{
    *paced = g_paced.load(std::memory_order_relaxed);
    *largest = g_largest.load(std::memory_order_relaxed);
    *last = g_last.load(std::memory_order_relaxed);
}

extern "C" float sfizz_release_pacing_start_cost()
{
    return g_measured.load(std::memory_order_relaxed);
}

extern "C" void sfizz_release_pacing_strikes(int* outcomes)
{
    for (int i = 0; i < kOutcomes; ++i)
        outcomes[i] = g_strikes[i].load(std::memory_order_relaxed);
}

namespace sfz {

ReleasePacingConfig releasePacingConfig() noexcept
{
    const Measure& m = g_measure;
    const float measured = m.denominator > 0.0 ? static_cast<float>(m.numerator / m.denominator) : 0.0f;
    return { g_cap.load(std::memory_order_relaxed), g_target.load(std::memory_order_relaxed),
             measured > 0.0f ? measured : g_startCost.load(std::memory_order_relaxed) };
}

ReleasePacingTick releasePacingTick() noexcept
{
    return g_tick;
}

void releasePacingRecord(int n) noexcept
{
    g_last.store(n, std::memory_order_relaxed);
    if (n > 1)
        g_paced.fetch_add(1, std::memory_order_relaxed);
    if (n > g_largest.load(std::memory_order_relaxed))
        g_largest.store(n, std::memory_order_relaxed);
}

void releasePacingVoiceStarted(bool struck) noexcept
{
    if (g_tick.serial == 0)
        return;
    if (struck)
        ++g_measure.struck;
    else
        ++g_measure.other;
}

void releasePacingRendered() noexcept
{
    if (g_tick.serial == 0)
        return;
    g_measure.renderEnd = Clock::now();
    g_measure.rendered = true;
}

} // namespace sfz
