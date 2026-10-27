// SPDX-License-Identifier: BSD-2-Clause
// See ReleasePacing.h.

#include "ReleasePacing.h"
#include <algorithm>
#include <atomic>

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
}

extern "C" void sfizz_release_pacing_counts(int* paced, int* largest, int* last)
{
    *paced = g_paced.load(std::memory_order_relaxed);
    *largest = g_largest.load(std::memory_order_relaxed);
    *last = g_last.load(std::memory_order_relaxed);
}

namespace sfz {

ReleasePacingConfig releasePacingConfig() noexcept
{
    return { g_cap.load(std::memory_order_relaxed), g_target.load(std::memory_order_relaxed),
             g_startCost.load(std::memory_order_relaxed) };
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

} // namespace sfz
