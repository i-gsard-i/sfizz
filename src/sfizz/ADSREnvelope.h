// SPDX-License-Identifier: BSD-2-Clause

// This code is part of the sfizz library and is licensed under a BSD 2-clause
// license. You should have receive a LICENSE.md file along with the code.
// If not, contact the sfizz maintainers at https://github.com/sfztools/sfizz

#pragma once
#include "Region.h"
#include "MidiState.h"
#include "utility/LeakDetector.h"
#include <absl/types/span.h>
namespace sfz {
/**
 * @brief Describe an attack/delay/sustain/release envelope that can
 * produce its coefficient in a blockwise manner for SIMD-type operations.
 */

class ADSREnvelope {
public:
    using Float = float;

    ADSREnvelope(const MidiState& state, CurveSet& curveSet)
    : midiState_(state), curveSet_(curveSet) {}
    /**
     * @brief Resets the ADSR envelope given a Region, the current midi state, and a delay and
     * trigger velocity
     *
     * @param desc
     * @param region
     * @param state
     * @param delay
     * @param velocity
     */
    void reset(const EGDescription& desc, const Region& region, int delay, float velocity, float sampleRate) noexcept;
    /**
     * @brief Get the next block of values for the envelope.
     *
     * @param output
     */
    void getBlock(absl::Span<Float> output) noexcept;
    /**
     * @brief Set the release time for the envelope
     *
     * @param timeInSeconds
     */
    void setReleaseTime(Float timeInSeconds) noexcept;
    /**
     * @brief Use a LINEAR release ramp (value -> 0 over the given time) instead of the
     * default exponential one. Used by the phase-aligned release-tail crossfade
     * (ReleaseAlign.h): a linear fade-out, paired with the release voice's linear fade-in
     * of the same duration, sums to constant amplitude for two in-phase copies (no bump,
     * no mid-crossfade dip). Pair with stopDynamicUpdates() so it is not re-derived.
     */
    void setLinearReleaseTime(Float timeInSeconds) noexcept;
    /**
     * @brief Override the (linear) attack ramp duration. Used to stretch the release
     * voice's fade-in to match the sustain's crossfade-out. Pair with stopDynamicUpdates().
     */
    void setAttackTime(Float timeInSeconds) noexcept;
    /**
     * @brief Start the envelope release after a delay.
     *
     * @param releaseDelay the delay before releasing in samples
     */
    void startRelease(int releaseDelay) noexcept;
    /**
     * @brief Cancel a release and get back into sustain.
     *
     * @param delay
     */
    void cancelRelease(int delay) noexcept;
    /**
     * @brief Is the envelope smoothing?
     *
     * @return true
     * @return false
     */
    bool isSmoothing() const noexcept { return currentState != State::Done; }
    /**
     * @brief Is the envelope released?
     *
     * @return true
     * @return false
     */
    bool isReleased() const noexcept { return currentState >= State::Release || shouldRelease; }
    /**
     * @brief Get the remaining delay samples
     *
     * @return int
     */
    int getRemainingDelay() const noexcept { return delay; }

    /**
     * @brief Stop dynamic updates of the EG values.
     *
     */
    void stopDynamicUpdates() { dynamic_ = false; }

    /**
     * @brief Shape the linear attack and the linear release as an equal-power crossfade
     * (composite stops, Composite.h): the attack ramp x is output as sin(pi/2 x), the
     * linear release from its starting level v0 as v0 sin(pi/2 x/v0). A summed tail is
     * not phase-aligned with the composite it replaces, and with phases that are
     * effectively random the expected power of a·s + b·t is a² + b², so a² + b² = 1 keeps
     * the level flat where a linear crossfade dips 3 dB. Set after setAttackTime() /
     * setLinearReleaseTime(); reset() clears it.
     */
    void setEqualPower(bool on) noexcept { equalPower_ = on; }

    /**
     * @brief Take over another envelope's running state (composite handover,
     * Composite.h): a member pipe started at the handover continues the composite's
     * envelope exactly where it is. Both must have been reset from the same description;
     * this envelope keeps its own description pointer.
     */
    void copyStateFrom(const ADSREnvelope& other) noexcept;

private:
    float sampleRate { config::defaultSampleRate };
    int secondsToSamples(Float timeInSeconds) const noexcept;
    Float secondsToLinRate(Float timeInSeconds) const noexcept;
    Float secondsToExpRate(Float timeInSeconds) const noexcept;
    void updateValues(int delay = 0) noexcept;
    void getBlockInternal(absl::Span<Float> output) noexcept;

    enum class State {
        Delay,
        Attack,
        Hold,
        Decay,
        Sustain,
        Release,
        Fadeout,
        Done
    };
    State currentState { State::Done };
    Float currentValue { 0.0 };
    const EGDescription* desc_ { nullptr };
    const MidiState& midiState_;
    CurveSet& curveSet_;
    float triggerVelocity_ { 0.0f };
    bool dynamic_ { false };
    int delay { 0 };
    Float attackStep { 0 };
    Float decayRate { 0 };
    Float releaseRate { 0 };
    Float releaseStep { 0 };          // per-sample decrement when linearRelease is set
    bool linearRelease { false };     // linear (vs exponential) release ramp
    int hold { 0 };
    Float start { 0 };
    Float sustain { 0 };
    Float sustainThreshold { config::virtuallyZero };
    int releaseDelay { 0 };
    bool shouldRelease { false };
    bool freeRunning { false };
    Float transitionDelta {};
    bool equalPower_ { false };       // equal-power shape on the linear ramps (composite)
    Float releaseFrom_ { 1 };         // the linear release's starting level
    LEAK_DETECTOR(ADSREnvelope);
};

}
