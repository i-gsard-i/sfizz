// SPDX-License-Identifier: BSD-2-Clause
//
// Division-tagged notes (Jukkasjarvi manual assignment; docs/design/MANUAL_ASSIGNMENT_DESIGN.md).
//
// Problem: sfizz has one global 128-note space and no channel concept, so a note event
// sounds every drawn stop. A virtual pipe organ needs notes routed to ONE console manual
// (pedalboard -> Pedal, keyboard A -> Manual I, a split keyboard -> two manuals) while
// keeping the zero-setup default where every input plays everything.
//
// Fix: regions carry a division bitmask (custom `division=N` opcode, 0-based index ->
// bit N; absent -> all bits set), and each note on/off event carries a mask. A region is
// eligible for a note event iff (region.divMask & event mask) != 0. The Voice records the
// note-on event's mask, so a note-off only releases voices of ITS division -- the same
// key held on two manuals releases independently. Default mask = all bits, so an engine
// that never sets a mask (or an SFZ with no division= tags) is byte-identical to stock.
//
// Transport: rather than change the public noteOn/noteOff signatures through the facade,
// the engine sets a "next event" mask immediately before each call. This is safe because
// sfz::Sfizz::noteOn dispatches synchronously on the calling (render) thread; the mask is
// consumed by that dispatch and resets to all-bits.

#pragma once
#include <cstdint>

extern "C" {
// Set the division mask consumed by the NEXT noteOn/noteOff dispatched on the same
// thread. Call immediately before each note call; unset events default to all-bits.
void sfizz_division_next_mask(uint32_t mask);
}

namespace sfz {

constexpr uint32_t kDivisionAll = 0xFFFFFFFFu;

// Return the pending event mask and reset it to all-bits.
uint32_t divisionConsumeMask() noexcept;

} // namespace sfz
