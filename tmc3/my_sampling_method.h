#ifndef MY_SAMPLING_METHOD_H
#define MY_SAMPLING_METHOD_H

#include "PCCTMC3Common.h"
#include <algorithm>
#include <iterator>
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

// edge case of swapArrElem
inline void swapArrElem(size_t i, size_t j) {
  return;
}

// swap arr[i] & arr[j], for each array in arrs, also do the same swapping;
template<typename ArrT, typename... ArrTs>
inline void swapArrElem(size_t i, size_t j, ArrT& arr, ArrTs&... arrs) {
  using std::swap;
  swap(arr[i], arr[j]);
  swapArrElem(i, j, arrs...);
}

inline void
mySamplingMethod(
  const Vec3<int32_t>& anchor,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& retained,
  const std::vector<int32_t>& neighborIndexes,
  int32_t lodIndex,
  int32_t (&localIndexes)[3],
  int64_t (&minDistances)[3])
{
  assert(neighborIndexes.size() <= 64);

  using std::vector;
  
  // Model Parameters
  const int DIST_W = 1;
  const int ANGLE_W = lodIndex + 1;
  const int MIN_CANDI = 3;

  // fall back onto KNN if too few points
  if (neighborIndexes.size() <= MIN_CANDI) {
    knnSamplingMethod(
      anchor, packedVoxel, retained, neighborIndexes, localIndexes,
      minDistances);
    return;
  }

  auto neigSize = neighborIndexes.size();
  vector<int> cost;
  vector<Vec3<int32_t>> pos;
  cost.reserve(neigSize);
  pos.reserve(neigSize);

  // translate the point, let the anchor point becomes origin point
  std::transform(neighborIndexes.begin(), neighborIndexes.end(), std::back_inserter(pos), [&](int k){
    Vec3<int32_t> pt = packedVoxel[retained[k]].bposition;
    return pt - anchor;
  });

  // initialize cost as distance
  std::transform(pos.begin(), pos.end(), std::back_inserter(cost), [&](Vec3<int32_t>& pt) {
    return pt.getNorm1();
  });
}
}  // namespace pcc

#endif  // MY_SAMPLING_METHOD_H