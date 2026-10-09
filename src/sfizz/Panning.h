#pragma once
#include "absl/types/span.h"
#include "MathHelpers.h"

namespace sfz
{

/**
* @brief Lookup a value from the pan table
*        No check is done on the range, needs to be capped
*        between 0 and panSize.
*
* @param pan
* @return float
*/
float panLookup(float pan);


/**
 * @brief Pans a mono signal left or right
 *
 * @param panEnvelope
 * @param leftBuffer
 * @param rightBuffer
 * @param size
 */
void pan(const float* panEnvelope, float* leftBuffer, float* rightBuffer, unsigned size) noexcept;
inline void pan(absl::Span<const float> panEnvelope, absl::Span<float> leftBuffer, absl::Span<float> rightBuffer) noexcept
{
    CHECK_SPAN_SIZES(panEnvelope, leftBuffer, rightBuffer);
    pan(panEnvelope.data(), leftBuffer.data(), rightBuffer.data(), minSpanSize(panEnvelope, leftBuffer, rightBuffer));
}

/**
 * @brief Controls the width of a stereo signal, setting it to mono when width = 0 and inverting the channels
 * when width = -1. Width = 1 has no effect.
 *
 * @param widthEnvelope
 * @param leftBuffer
 * @param rightBuffer
 * @param size
 */
void width(const float* widthEnvelope, float* leftBuffer, float* rightBuffer, unsigned size) noexcept;
inline void width(absl::Span<const float> widthEnvelope, absl::Span<float> leftBuffer, absl::Span<float> rightBuffer) noexcept
{
    CHECK_SPAN_SIZES(widthEnvelope, leftBuffer, rightBuffer);
    width(widthEnvelope.data(), leftBuffer.data(), rightBuffer.data(), minSpanSize(widthEnvelope, leftBuffer, rightBuffer));
}

/**
 * @brief A stereo voice's pan, width and position stages for values constant over a block,
 * and a gain, folded into one matrix: L' = m[0] L + m[1] R, R' = m[2] L + m[3] R.
 * It takes the table entries pan() and width() take on their vector path, so it gives
 * their result up to float rounding, for one multiply-add pair per frame instead of three
 * table passes and two gain passes.
 */
void stereoMatrix(float pan, float width, float position, float gain, float m[4]) noexcept;

}
