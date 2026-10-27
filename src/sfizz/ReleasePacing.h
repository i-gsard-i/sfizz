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
// voices sounding, capped. T = B/C - 1, B the stream's buffer and C its callback in frames:
// a callback is woken when there is room for C frames, so it starts with B - C still queued,
// and one that renders longer than that runs the queue dry. A release that fits gets n = 1 and goes as before. startCost is
// measured live from the chords struck (ReleasePacing.cpp), the configured value until then.
//
// Only plain note-offs are paced (no stop filter), and only on a thread whose engine
// reports its callbacks: the composite renderer's synth, on its own thread, never paces.
#pragma once
#include <cstdint>

extern "C" {
// cap <= 1 turns pacing off. target: the load a release may reach, as a share of the
// callback; <= 0: from the stream's buffer (B/C - 1, the default). startCost: a voice starting, against one sounding, until the engine has
// measured it (1.28 on the PC, build16).
void sfizz_release_pacing(int cap, float target, float startCost);
// From the engine at the start of each callback, before its events: the callback's frames,
// its load (a share of the callback, smoothed) and the stream's buffer in frames (0: not
// known). Per thread, as the division mask.
void sfizz_release_pacing_callback(int frames, float load, int bufferFrames);
// Since the start: releases paced (n > 1), the largest n, and the last n chosen.
void sfizz_release_pacing_counts(int* paced, int* largest, int* last);
// The last release the model judged: its held load L, its starts' load m * startCost * L / V,
// the callback's frames, and its slowest callback (s, from its report to its last render's
// end) in the 40 callbacks from it (to check the target against the buffer).
void sfizz_release_pacing_last(float* load, float* starts, int* frames, float* peak);
// The start cost measured on the reporting thread (0 until measured).
float sfizz_release_pacing_start_cost();
// What became of the strikes it is measured from: {measured, not while playing,
// ended by other starts, no rise after (in the estimate all the same)}.
void sfizz_release_pacing_strikes(int* outcomes /* [4] */);
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
    int bufferFrames;
};
ReleasePacingTick releasePacingTick() noexcept;

void releasePacingRecord(int n, float load = 0.0f, float starts = 0.0f, int frames = 0) noexcept;
// For the start cost measured live (ReleasePacing.cpp): a voice started (struck: by a
// note-on), and a render's end with the voices still active. Ignored on a thread whose
// callbacks are not reported.
void releasePacingVoiceStarted(bool struck) noexcept;
void releasePacingRendered(int activeVoices) noexcept;

} // namespace sfz
