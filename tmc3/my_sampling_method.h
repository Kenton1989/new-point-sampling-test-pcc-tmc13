#ifndef MY_SAMPLING_METHOD_H
#define MY_SAMPLING_METHOD_H

#include "PCCTMC3Common.h"
#include <vector>

namespace pcc {

// Original sampling method
inline void
knnSamplingMethod(
  const Vec3<int32_t>& anchor,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& retained,
  const std::vector<int32_t>& neighborIndexes,
  int32_t (&localIndexes)[3],
  int64_t (&minDistances)[3])
{
  for (const auto k : neighborIndexes) {
    updateNearestNeigh(
      anchor, packedVoxel[retained[k]].bposition, k, localIndexes,
      minDistances);
  }
}

inline void
mySamplingMethod(
  const Vec3<int32_t>& anchor,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& retained,
  const std::vector<int32_t>& neighborIndexes,
  int32_t (&localIndexes)[3],
  int64_t (&minDistances)[3])
{
  assert(neighborIndexes.size() <= 64);

  // fall back onto KNN if too few points
  if (neighborIndexes.size() <= 3) {
    knnSamplingMethod(
      anchor, packedVoxel, retained, neighborIndexes, localIndexes,
      minDistances);
  }

  using std::vector;

  // vector<int> cost
}
}  // namespace pcc

#endif  // MY_SAMPLING_METHOD_H