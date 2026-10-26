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
// eligible for a note event iff (region.divMask & event mask) != 0. The Voice records
// the INTERSECTION (event mask & region divMask) -- the divisions it actually sounds
// for -- so a note-off only releases voices of ITS division(s): the same key held on
// two manuals releases independently, and a coupler retro-release of just the source
// division leaves the destination's own voices sounding. Default mask = all bits, so an
// engine that never sets a mask (or an SFZ with no division= tags) is byte-identical.
//
// Transport: rather than change the public noteOn/noteOff signatures through the facade,
// the engine sets a "next event" mask immediately before each call. This is safe because
// sfz::Sfizz::noteOn dispatches synchronously on the calling (render) thread; the mask is
// consumed by that dispatch and resets to all-bits.
//
// Stop gate-CC filter (Jukkasjarvi live registration; LIVE_REGISTRATION_DESIGN.md): the
// same next-event transport also carries an optional stop filter. Regions carry their
// stop's gate CC (custom `stop_cc=N` opcode, absent -> -1 = untagged); when an event
// carries a filter >= 0, it acts ONLY on regions/voices of that stop. This lets the
// engine attack or release ONE stop's pipes for an already-held key when the organist
// draws/retires a stop mid-chord. Default -1 = no filter = byte-identical to stock.

#pragma once
#include <cstdint>

extern "C" {
// Set the division mask consumed by the NEXT noteOn/noteOff dispatched on the same
// thread. Call immediately before each note call; unset events default to all-bits.
void sfizz_division_next_mask(uint32_t mask);
// Set the stop gate-CC filter consumed by the NEXT noteOn/noteOff on the same thread:
// the event triggers/releases only regions whose stop_cc == cc. -1 (default) = all.
void sfizz_stop_next_filter(int cc);
}

namespace sfz {

constexpr uint32_t kDivisionAll = 0xFFFFFFFFu;
constexpr int kStopFilterNone = -1;
// A registration change on held notes (Composite.h, "Restrike"): two special filters.
// On a note-off: release only the note's composites, each into its summed tail, and mark
// the note to be struck again on those divisions. On a note-on: strike the marked note
// again with the registration drawn now, leaving the stops still sounding on it alone.
constexpr int kStopFilterCompositeRelease = -2;
constexpr int kStopFilterRestrike = -3;
// Octave couplers (Composite.h): on a note-on, strike the key and its coupled rows as the
// division's octave composite, or as plain notes on each row where it has none; on a
// note-off, let the key's octave composites go whole, into their summed tails.
constexpr int kStopFilterOctave = -4;

// Return the pending event mask and reset it to all-bits.
uint32_t divisionConsumeMask() noexcept;

// Return the pending stop filter and reset it to none.
int stopConsumeFilter() noexcept;

} // namespace sfz
