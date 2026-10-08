// SPDX-License-Identifier: BSD-2-Clause

// This code is part of the sfizz library and is licensed under a BSD 2-clause
// license. You should have receive a LICENSE.md file along with the code.
// If not, contact the sfizz maintainers at https://github.com/sfztools/sfizz

#pragma once
#include "AudioSpan.h"
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
};

} // namespace sfz
