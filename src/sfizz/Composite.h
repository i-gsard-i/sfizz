// SPDX-License-Identifier: BSD-2-Clause
//
// Composite stops (organdroid docs/design/COMPOSITE_STOPS_DESIGN.md).
//
// Problem: rapid re-strikes start N pipe voices per key per manual while the previous
// chord's fades still sound, and that churn is where a phone runs out of CPU. Held chords
// cost little.
//
// Fix: for one manual's drawn stops, a pre-summed attack per key (a "composite region",
// `composite=N`) plays as ONE voice for the first T_h of the note. Mixing is linear, so
// the sum, rendered beforehand by the same engine, sounds the same as the N pipes. At T_h
// (or when a stop is retired from the held note) the composite HANDS OVER: it stops, and
// each member pipe starts at the frame it would have reached, with the composite's
// envelope and LFO state, so the note goes on as ordinary live pipes. A note released
// before T_h fades into ONE summed tail (a `composite=N trigger=release` region) through
// an equal-power crossfade, scaled to the note's level as each pipe's tail is.
//
// Lookup is by the manual's drawn set, which only the app knows: it activates composite N
// for a division when that division's drawn stops are exactly N's members, and
// deactivates it (-1) when they change. Activation goes through the app's event queue,
// so it is ordered with the note events around it. Stops added to a held note are started
// live as today; a stop retired from a held note (a stop-filtered note-off) hands that
// note's composite over first, so the retired stop's own voice can be released.
//
// Restrike (composite mode, a piston on held notes): a composite only lives for the first
// T_h of its note, so a registration change then has two cases. Past T_h the note is live
// pipes, and live registration applies. Before it, handing over would start every member
// pipe; instead the composite is released into its summed tail (a kStopFilterCompositeRelease
// note-off), and once the app has set the new registration, the note is struck again with
// it (a kStopFilterRestrike note-on): the new composite, or its stops live. Stops still
// sounding on the note (a stop that cannot be summed, drawn in both registrations) are left
// alone, and a stop-filtered attack on a note waiting to be struck again is dropped, since
// the strike will start that stop.
//
// Inert by default: no division has an active composite, composite regions never trigger
// through normal dispatch, and the output is byte-identical to an engine without them.

#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
// Make composite `id` (the `composite=` value of its regions) the one new notes of
// console division `division` (0-based, the `division=` index) use; -1 = none.
void sfizz_composite_activate(int division, int id);
// Deactivate every division.
void sfizz_composite_clear();
}

namespace sfz {

// Most member pipes one composite can hand over to (stops of one manual).
constexpr size_t kMaxCompositeMembers = 64;

// The composite active for division `division` (0..31), -1 = none.
int compositeActive(int division) noexcept;

// True when some division has an active composite (a cheap guard for the note path).
bool compositeAnyActive() noexcept;

} // namespace sfz
