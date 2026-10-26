// SPDX-License-Identifier: BSD-2-Clause
// See Composite.h.

#include "Composite.h"
#include <atomic>

namespace {
// Written by the app's event path (the render thread, between blocks) and read by the
// note-on dispatch on the same thread. Atomic anyway, as in DivisionGate.cpp.
std::atomic<int> g_active[32] = {
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
};
std::atomic<uint32_t> g_activeMask { 0 };
std::atomic<int> g_octave[32] = {
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
};
}

extern "C" void sfizz_composite_activate_octave(int division, int id)
{
    if (division < 0 || division >= 32)
        return;
    g_octave[division].store(id < 0 ? -1 : id, std::memory_order_relaxed);
}

extern "C" void sfizz_composite_activate(int division, int id)
{
    if (division < 0 || division >= 32)
        return;
    g_active[division].store(id < 0 ? -1 : id, std::memory_order_relaxed);
    const uint32_t bit = 1u << division;
    if (id < 0)
        g_activeMask.fetch_and(~bit, std::memory_order_relaxed);
    else
        g_activeMask.fetch_or(bit, std::memory_order_relaxed);
}

extern "C" void sfizz_composite_clear()
{
    for (auto& a : g_active)
        a.store(-1, std::memory_order_relaxed);
    for (auto& a : g_octave)
        a.store(-1, std::memory_order_relaxed);
    g_activeMask.store(0, std::memory_order_relaxed);
}

namespace sfz {

int compositeActive(int division) noexcept
{
    if (division < 0 || division >= 32)
        return -1;
    return g_active[division].load(std::memory_order_relaxed);
}

bool compositeAnyActive() noexcept
{
    return g_activeMask.load(std::memory_order_relaxed) != 0;
}

int compositeOctaveActive(int division) noexcept
{
    if (division < 0 || division >= 32)
        return -1;
    return g_octave[division].load(std::memory_order_relaxed);
}

} // namespace sfz
