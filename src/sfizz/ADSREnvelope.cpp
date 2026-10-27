// SPDX-License-Identifier: BSD-2-Clause

// This code is part of the sfizz library and is licensed under a BSD 2-clause
// license. You should have receive a LICENSE.md file along with the code.
// If not, contact the sfizz maintainers at https://github.com/sfztools/sfizz

#include "ADSREnvelope.h"
#include "Config.h"
#include "SIMDHelpers.h"
#include "MathHelpers.h"

namespace sfz {

using Float = ADSREnvelope::Float;

int ADSREnvelope::secondsToSamples(Float timeInSeconds) const noexcept
{
    if (timeInSeconds <= 0)
        return Float(0);

    return static_cast<int>(timeInSeconds * sampleRate);
};

Float ADSREnvelope::secondsToLinRate(Float timeInSeconds) const noexcept
{
    if (timeInSeconds <= 0)
        return Float(1);

    return 1 / (sampleRate * timeInSeconds);
};

Float ADSREnvelope::secondsToExpRate(Float timeInSeconds) const noexcept
{
    if (timeInSeconds <= 0)
        return Float(0.0);

    timeInSeconds = std::max(Float(Default::offTime), timeInSeconds);
    return std::exp(Float(-9.0) / (timeInSeconds * sampleRate));
};

void ADSREnvelope::reset(const EGDescription& desc, const Region& region, int delay, float velocity, float sampleRate) noexcept
{
    this->sampleRate = sampleRate;
    desc_ = &desc;
    dynamic_ = desc.dynamic;
    triggerVelocity_ = velocity;
    currentState = State::Delay; // Has to be before the update
    updateValues(delay);
    releaseDelay = 0;
    shouldRelease = false;
    linearRelease = false;
    equalPower_ = false;
    freeRunning = (
        (this->sustain <= Float(config::sustainFreeRunningThreshold))
        || (region.loopMode == LoopMode::one_shot && region.isOscillator())
    );
    currentValue = this->start;
}

void ADSREnvelope::updateValues(int delay) noexcept
{
    if (currentState == State::Delay)
        this->delay = delay + secondsToSamples(desc_->getDelay(midiState_, curveSet_, triggerVelocity_, delay));
    this->attackStep = secondsToLinRate(desc_->getAttack(midiState_, curveSet_, triggerVelocity_, delay));
    this->decayRate = secondsToExpRate(desc_->getDecay(midiState_, curveSet_, triggerVelocity_, delay));
    this->releaseRate = secondsToExpRate(desc_->getRelease(midiState_, curveSet_, triggerVelocity_, delay));
    this->hold = secondsToSamples(desc_->getHold(midiState_, curveSet_, triggerVelocity_, delay));
    this->sustain = clamp(desc_->getSustain(midiState_, curveSet_, triggerVelocity_, delay), 0.0f, 1.0f);
    this->start = clamp(desc_->getStart(midiState_, curveSet_, triggerVelocity_, delay), 0.0f, 1.0f);
    sustainThreshold = this->sustain + config::virtuallyZero;
}

void ADSREnvelope::getBlock(absl::Span<Float> output) noexcept
{
    if (dynamic_) {
        int processed = 0;
        int remaining = static_cast<int>(output.size());
        while(remaining > 0) {
            updateValues(processed);
            int chunkSize = min(config::processChunkSize, remaining);
            getBlockInternal(output.subspan(processed, chunkSize));
            processed += chunkSize;
            remaining -= chunkSize;
        }
    } else {
        getBlockInternal(output);
    }
}

void ADSREnvelope::getBlockInternal(absl::Span<Float> output) noexcept
{
    State currentState = this->currentState;
    Float currentValue = this->currentValue;
    bool shouldRelease = this->shouldRelease;
    int releaseDelay = this->releaseDelay;
    Float transitionDelta = this->transitionDelta;

    while (!output.empty()) {
        size_t count = 0;
        size_t size = output.size();

        if (shouldRelease) {
            if (releaseDelay > 0) {
                // prevent computing the segment further than release point
                size = std::min<size_t>(size, releaseDelay);
            } else if (releaseDelay == 0 && delay <= 0) {
                if (delay < 0) {
                    // release takes effect this frame
                    currentState = State::Release;
                } else {
                    // release takes effect the next frame
                    size = 1;
                }
                releaseDelay = -1;
            } else if (releaseDelay == -1 && currentState < State::Release && delay <= 0) {
                // release takes effect this frame
                currentState = State::Release;
            }
        }

        Float previousValue;

        switch (currentState) {
        case State::Delay:
            while (count < size && delay-- > 0) {
                currentValue = start;
                output[count++] = currentValue;
            }
            if (delay <= 0)
                currentState = State::Attack;
            break;
        case State::Attack:
            if (equalPower_) {
                while (count < size && (currentValue += attackStep) < 1)
                    output[count++] = std::sin(Float(1.5707963267948966) * currentValue);
            } else {
                while (count < size && (currentValue += attackStep) < 1)
                    output[count++] = currentValue;
            }
            if (currentValue >= 1) {
                currentValue = 1;
                currentState = State::Hold;
            }
            break;
        case State::Hold:
            while (count < size && hold-- > 0)
                output[count++] = currentValue;
            if (hold <= 0)
                currentState = State::Decay;
            break;
        case State::Decay:
            while (count < size && (currentValue *= decayRate) > sustain)
                output[count++] = currentValue;
            if (currentValue <= sustainThreshold) {
                currentState = State::Sustain;
                currentValue = std::max(sustain, currentValue);
                transitionDelta = (sustain - currentValue) / (sampleRate * config::egTransitionTime);
            }
            break;
        case State::Sustain:
            if (!shouldRelease && freeRunning) {
                shouldRelease = true;
                break;
            }
            while (count < size && currentValue > sustain) {
                currentValue += transitionDelta;
                output[count++] = currentValue;
            }
            // Settled on the sustain level (a held pipe): the rest of the segment is one
            // value, the same samples the per-sample loop wrote, as one vectorized fill.
            std::fill(output.begin() + count, output.begin() + size, currentValue);
            count = size;
            break;
        case State::Release:
            previousValue = currentValue;
            if (linearRelease && equalPower_) {
                const Float v0 = releaseFrom_;
                while (count < size && (currentValue -= releaseStep) > config::egReleaseThreshold)
                    output[count++] = previousValue = v0 * std::sin(Float(1.5707963267948966) * currentValue / v0);
            } else if (linearRelease) {
                while (count < size && (currentValue -= releaseStep) > config::egReleaseThreshold)
                    output[count++] = previousValue = currentValue;
            } else {
                while (count < size && (currentValue *= releaseRate) > config::egReleaseThreshold)
                    output[count++] = previousValue = currentValue;
            }
            if (currentValue <= config::egReleaseThreshold) {
                currentState = State::Fadeout;
                currentValue = previousValue;
                transitionDelta = -max(config::egReleaseThreshold, currentValue)
                    / (sampleRate * config::egTransitionTime);
            }
            break;
        case State::Fadeout:
            while (count < size && (currentValue += transitionDelta) > 0)
                output[count++] = currentValue;
            if (currentValue <= 0) {
                currentState = State::Done;
                currentValue = 0;
            }
            break;
        default:
            count = size;
            currentValue = 0.0;
            sfz::fill(output, currentValue);
            break;
        }

        if (shouldRelease)
            releaseDelay = std::max(-1, releaseDelay - static_cast<int>(count));

        output.remove_prefix(count);
    }

    this->currentState = currentState;
    this->currentValue = currentValue;
    this->shouldRelease = shouldRelease;
    this->releaseDelay = releaseDelay;
    this->transitionDelta = transitionDelta;

    ASSERT(!hasNanInf(output));
}

void ADSREnvelope::startRelease(int releaseDelay) noexcept
{
    shouldRelease = true;
    this->releaseDelay = releaseDelay;
}

void ADSREnvelope::cancelRelease(int delay) noexcept
{
    (void)delay;
    currentState = State::Sustain;
    shouldRelease = false;
    this->releaseDelay = -1;
}

int ADSREnvelope::releaseFramesLeft() const noexcept
{
    if (!isReleased())
        return -1;
    const Float v = std::max(currentValue, config::egReleaseThreshold);
    Float left = 0;
    if (linearRelease)
        left = releaseStep > 0 ? (v - config::egReleaseThreshold) / releaseStep : 0;
    else if (releaseRate > 0)
        left = std::log(config::egReleaseThreshold / v) / std::log(releaseRate);
    const int delayed = shouldRelease && releaseDelay > 0 ? releaseDelay : 0;
    return delayed + static_cast<int>(std::ceil(left));
}

void ADSREnvelope::setReleaseTime(Float timeInSeconds) noexcept
{
    releaseRate = secondsToExpRate(timeInSeconds);
}

void ADSREnvelope::setLinearReleaseTime(Float timeInSeconds) noexcept
{
    linearRelease = true;
    // Ramp from the CURRENT value to 0 over timeInSeconds, so the fade takes the full
    // crossfade window regardless of how far the attack had progressed (short notes
    // release mid-onset at <1.0). Keeps it complementary to the release voice's fade-in.
    const Float samples = std::max(Float(1), sampleRate * timeInSeconds);
    releaseFrom_ = std::max(currentValue, config::egReleaseThreshold);
    releaseStep = releaseFrom_ / samples;
}

void ADSREnvelope::copyStateFrom(const ADSREnvelope& other) noexcept
{
    sampleRate = other.sampleRate;
    currentState = other.currentState;
    currentValue = other.currentValue;
    triggerVelocity_ = other.triggerVelocity_;
    dynamic_ = other.dynamic_;
    delay = other.delay;
    attackStep = other.attackStep;
    decayRate = other.decayRate;
    releaseRate = other.releaseRate;
    releaseStep = other.releaseStep;
    linearRelease = other.linearRelease;
    hold = other.hold;
    start = other.start;
    sustain = other.sustain;
    sustainThreshold = other.sustainThreshold;
    releaseDelay = other.releaseDelay;
    shouldRelease = other.shouldRelease;
    freeRunning = other.freeRunning;
    transitionDelta = other.transitionDelta;
    equalPower_ = other.equalPower_;
    releaseFrom_ = other.releaseFrom_;
}

void ADSREnvelope::setAttackTime(Float timeInSeconds) noexcept
{
    attackStep = secondsToLinRate(timeInSeconds);
}

}
