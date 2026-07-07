// SPDX-License-Identifier: BSD-2-Clause
// See DivisionGate.h.

#include "DivisionGate.h"
#include <atomic>

namespace {
// Producer and consumer are the same (render) thread -- the engine sets the mask right
// before calling noteOn/noteOff, which dispatches synchronously. Atomic anyway: it is
// cheap, and it keeps a stray off-thread caller merely harmless instead of UB.
std::atomic<uint32_t> g_nextMask { sfz::kDivisionAll };
std::atomic<int>      g_nextStop { sfz::kStopFilterNone };
}

extern "C" void sfizz_division_next_mask(uint32_t mask)
{
    g_nextMask.store(mask, std::memory_order_relaxed);
}

extern "C" void sfizz_stop_next_filter(int cc)
{
    g_nextStop.store(cc, std::memory_order_relaxed);
}

namespace sfz {

uint32_t divisionConsumeMask() noexcept
{
    return g_nextMask.exchange(kDivisionAll, std::memory_order_relaxed);
}

int stopConsumeFilter() noexcept
{
    return g_nextStop.exchange(kStopFilterNone, std::memory_order_relaxed);
}

} // namespace sfz
