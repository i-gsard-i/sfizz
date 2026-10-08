// SPDX-License-Identifier: BSD-2-Clause
// See DivisionGate.h.

#include "DivisionGate.h"

namespace {
// Producer and consumer are the same thread -- the engine sets the mask right before
// calling noteOn/noteOff, which dispatches synchronously. Per thread, so a second synth
// on another thread (the composite renderer) never takes the render thread's pending
// mask, nor the render thread its own.
thread_local uint32_t g_nextMask { sfz::kDivisionAll };
thread_local int      g_nextStop { sfz::kStopFilterNone };
}

extern "C" void sfizz_division_next_mask(uint32_t mask)
{
    g_nextMask = mask;
}

extern "C" void sfizz_stop_next_filter(int cc)
{
    g_nextStop = cc;
}

namespace sfz {

uint32_t divisionConsumeMask() noexcept
{
    const uint32_t mask = g_nextMask;
    g_nextMask = kDivisionAll;
    return mask;
}

int stopConsumeFilter() noexcept
{
    const int cc = g_nextStop;
    g_nextStop = kStopFilterNone;
    return cc;
}

} // namespace sfz
