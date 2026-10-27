// SPDX-License-Identifier: BSD-2-Clause
//
// Release pacing (Jukkasjarvi organ; docs/tests/RELEASE_PACING_TEST_REPORT.md).
//
// At a note-off every held pipe starts its release tail and fades out over the crossfade,
// so the callback that takes a big release renders the held chord and its tails at once
// and starts every tail: about L + A, with L the held chord's load and A the load of
// starting it. Split by stop into n groups, each released one crossfade after the last,
// each callback carries L + A/n. So when the load the engine reports says a release would
// pass the target T, the plain note-offs of one callback are decided together at its
// render: n = ceil(m * startCost * (L / V) / (T - L)), m the voices they release and V the
// voices sounding, capped. A release that fits gets n = 1 and goes as before.
//
// Only plain note-offs are paced (no stop filter), and only on a thread whose engine
// reports its callbacks: the composite renderer's synth, on its own thread, never paces.
#pragma once
#include <cstdint>

extern "C" {
// cap <= 1 turns pacing off. target: the load a release may reach, as a share of the
// callback. startCost: a voice starting, against one sounding (1.28 on the PC, build16).
void sfizz_release_pacing(int cap, float target, float startCost);
// From the engine at the start of each callback, before its events: the callback's frames
// and its load (a share of the callback, smoothed). Per thread, as the division mask.
void sfizz_release_pacing_callback(int frames, float load);
// Since the start: releases paced (n > 1), the largest n, and the last n chosen.
void sfizz_release_pacing_counts(int* paced, int* largest, int* last);
}

namespace sfz {

struct ReleasePacingConfig {
    int cap;
    float target;
    float startCost;
};
ReleasePacingConfig releasePacingConfig() noexcept;

struct ReleasePacingTick {
    uint32_t serial;   // 0: this thread's callbacks are not reported, no pacing
    int frames;
    float load;
};
ReleasePacingTick releasePacingTick() noexcept;

void releasePacingRecord(int n) noexcept;

} // namespace sfz
