// SPDX-License-Identifier: BSD-2-Clause
//
// Phase-aligned release tails (Jukkasjarvi "A2" plan; docs/RELEASE_TAIL_REPORT.md).
//
// Problem: sfizz starts every trigger=release sample at frame 0, at a RANDOM phase vs
// the running sustain loop. Summing that decorrelated full-level copy of the same tone
// onto the fading sustain produces an audible note-off "dab" (comb/+6 dB bump). GrandOrgue
// avoids it by phase-aligning the release to the loop.
//
// Fix: a tiny global registry mapping an ATTACK sample (the recording whose live loop
// position we can read exactly via Voice::getSourcePosition()) to its precomputed
// {period P, anchor delta0}, where delta0 is the release frame phase-aligned with the
// loop's loop_start (computed offline by tools/align.py -> release_align.tsv). At note-off
// Synth reads the dying attack voice's phase phi = (sourcePosition - loopStart) mod P and
// starts the release at O = delta0 + phi (mod P), so the handover is between two in-phase
// copies of the tone. Keyed by the ATTACK sample because that is where phi is read.
//
// Lifetime/threading: populated once at load via the extern "C" API (engine thread, before
// audio); read on the audio thread at note-off ONLY (a few times/sec). The map is never
// mutated while audio runs, so lock-free reads are safe; the enable flag is atomic.

#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

extern "C" {
// Register/replace one attack-sample alignment entry. period & anchor are in source frames;
// loopRms is the (L+R) RMS of the loop body (the pipe's full level reference for tail scaling).
void sfizz_release_align_set(const char* attackSample, float period, int anchor, float loopRms);
// Attach a phase track to an entry already set (docs/design/RELEASE_PHASE_TRACK_DESIGN.md).
// `column` is the tsv's sixth column as written by tools/phase_track.py: "v1:" + base64 of
// one byte per point, point i at source frame i*4*round(period), the byte being the
// correction to the anchor's start in 1/256 of a period. The anchor holds only near
// loop_start: a pipe recorded in a church drifts, so elsewhere the tail would start at a
// near-random phase. Returns the number of points, 0 if the column is not understood.
int sfizz_release_align_set_track(const char* attackSample, const char* column);
// Attach the attack's onset levels to an entry already set (docs/design/RESTRIKE_HANDOVER_DESIGN.md
// §4): the tsv's seventh column, "o1:" + base64 of one byte per period from frame 0, the byte
// being the (L+R) RMS over that period in 1/255 of loopRms, up to the first period that
// reaches it. Returns the number of periods, 0 if the column is not understood.
int sfizz_release_align_set_onset(const char* attackSample, const char* column);
// Attach the attack's tail track to an entry already set (RESTRIKE_HANDOVER_DESIGN.md §4): the
// tsv's eighth column, "t2:" + base64 of the first point's 3 bytes, then 2 per further point.
// Point i is at tail frame round(period)*i, its value the attack frame to enter at, in level and
// phase, when the tail is there at its recorded level; after the first, each is stored as its
// step from the one before minus round(period), signed 16-bit, little-endian. The track ends
// where the tail falls below 1/255 of loopRms. Returns the number of points, 0 if not understood.
int sfizz_release_align_set_tail(const char* attackSample, const char* column);
// Drop all entries (call before re-registering on a fresh organ load).
void sfizz_release_align_clear();
// Master enable, driven by the app's "Release tails" toggle. Default OFF -> byte-identical
// to the unaligned path. (The SFZ CC103 gate still independently controls whether a release
// region triggers at all; this only controls whether a triggered release is phase-aligned.)
void sfizz_release_align_enable(int on);
// Crossfade time (seconds) over which the sustain voice fades OUT as the phase-aligned
// release fades in. The whole point of alignment: the two in-phase copies overlap for this
// window so the sum stays ~constant (no +6 dB bump, no gap). Default 0.030.
void sfizz_release_align_set_crossfade(float seconds);
// Fallback fade time (seconds) for looped pipes that have NO recorded release tail (the
// lowest C of each rank): they get a plain exponential fade of this length instead of being
// cut at the crossfade window. Set to ~the typical recorded-tail length (align.py).
void sfizz_release_align_set_fallback(float seconds);
// "Cut release tails" (organdroid, docs/tests/VOICE_POLYPHONY_REPORT.md): every
// trigger=release voice is faded out so it is silent `seconds` after it starts. 0 = the
// recorded tails play whole (default). Live: read by each release voice as it renders.
void sfizz_release_tail_cut(float seconds);
// "A re-struck pipe ends its tail" (organdroid, docs/design/RESTRIKE_HANDOVER_DESIGN.md): one pipe
// gives one sound, so its new attack can end the tails its earlier strokes left ringing
// (same stop_cc, same key), over sfizz's fast off time. A tail is the pipe's dying speech
// plus, in a wet recording, the room's decay, which really goes on; the samples do not
// separate them. 0 = by organ (only a region with organdroid_wet=0), 1 = always,
// 2 = never (default 0). Live: read at each note-on.
void sfizz_restrike_ends_tails(int mode);
}

namespace sfz {

// Canonical lookup key = the WHOLE sample path, normalized to '/' separators. The two load
// modes each register and look up a consistent path form (legacy master = relative paths;
// multi-organ palette = absolute per-organ paths written into both sample= and the tsv), so
// exact matching works for both AND keeps keys organ-unique in a mixed palette (the absolute
// path carries the organ directory; a bare "Stop/file.flac" would collide across organs).
std::string releaseAlignKey(const std::string& path);

bool releaseAlignEnabled() noexcept;

// Sustain fade-out / crossfade time in seconds (see setter above).
float releaseAlignCrossfade() noexcept;

// Fallback fade time in seconds for no-tail pipes (see setter above).
float releaseAlignFallback() noexcept;

// Release-tail cut in seconds, 0 = none (see setter above).
float releaseTailCut() noexcept;

// Whether a re-strike of `tail`'s pipe ends it (see sfizz_restrike_ends_tails).
bool restrikeEndsTail(bool wet) noexcept;

// On hit, fills period/anchor (source frames) + loopRms and returns true. track/trackSize
// get the entry's phase track (nullptr/0 if it has none); valid until the next clear.
bool releaseAlignLookup(const std::string& attackSample, float& period, int& anchor, float& loopRms,
                        const uint8_t*& track, size_t& trackSize);

// Correction (source frames) the phase track adds at source position `pos`: the nearest
// point's byte * period / 256. 0 without a track.
float releaseAlignTrackCorrection(const uint8_t* track, size_t trackSize, float period, int64_t pos) noexcept;

// The attack's onset levels (see sfizz_release_align_set_onset); false without them.
bool releaseAlignOnset(const std::string& attackSample, const uint8_t*& onset, size_t& onsetSize);

// The source frame where the attack is first as loud as `level` (of its full level): the
// start of the first period whose onset level reaches it, or the end of the onset.
int64_t restrikeEntry(const uint8_t* onset, size_t onsetSize, float period, float level) noexcept;

// The attack's tail track (see sfizz_release_align_set_tail), in bytes; false without it.
bool releaseAlignTail(const std::string& attackSample, const uint8_t*& tail, size_t& tailSize);

// The attack frame to enter at when the tail is at frame `pos`: the nearest point's, plus the
// frames since it. -1 without points or past the last (the tail is then cut).
int64_t restrikeTailEntry(const uint8_t* tail, size_t tailSize, float period, int64_t pos) noexcept;

} // namespace sfz
