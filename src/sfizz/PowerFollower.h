// SPDX-License-Identifier: BSD-2-Clause

// This code is part of the sfizz library and is licensed under a BSD 2-clause
// license. You should have receive a LICENSE.md file along with the code.
// If not, contact the sfizz maintainers at https://github.com/sfztools/sfizz

#pragma once
#include "AudioSpan.h"
#include <array>
#include <memory>

namespace sfz {

class PowerFollower {
public:
    PowerFollower();
    void setSampleRate(float sampleRate) noexcept;
    void setSamplesPerBlock(unsigned samplesPerBlock);
    void process(AudioSpan<float> buffer) noexcept;
    void clear() noexcept;
    // Start from another follower's state scaled by `weight` (a composite member taking
    // over a share of the composite's power, Composite.h).
    void copyScaledFrom(const PowerFollower& other, float weight) noexcept;
    float getAveragePower() const noexcept { return currentPower_; }
    // Mean power of the last `steps` whole steps (at most kRecentSteps), unsmoothed: the level
    // a voice has now. The smoothed one lags a decaying tail by its 200 ms release.
    float getRecentPower(size_t steps) const noexcept;
    // 8 steps of 512 frames: one period of a 32' C (16.35 Hz) at up to 64 kHz.
    static constexpr size_t kRecentSteps { 8 };

private:
    void updateTrackingFactor() noexcept;

private:
    float sampleRate_ {};
    unsigned samplesPerBlock_ {};

    std::unique_ptr<float[]> tempBuffer_;

    float attackTrackingFactor_ {};
    float releaseTrackingFactor_ {};

    float currentPower_ {};
    float currentSum_ = 0;
    size_t currentCount_ = 0;
    std::array<float, kRecentSteps> recent_ {};
    size_t recentNext_ = 0;
    size_t recentFilled_ = 0;
};

} // namespace sfz
