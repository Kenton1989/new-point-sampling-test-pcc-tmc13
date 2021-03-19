#include "PCCTMC3Common.h"
#include "test_config.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace pcc {

namespace Kenton {

// edge case of swapArrElem below
inline void swapArrElem(size_t i, size_t j) {
  // do nothing
}

// swap arr[i] & arr[j]. And for each array in arrs, do the same swapping;
template<typename ArrT, typename... ArrTs>
inline void swapArrElem(size_t i, size_t j, ArrT& arr, ArrTs&... arrs) {
  using std::swap;
  swap(arr[i], arr[j]);
  swapArrElem(i, j, arrs...);
}

// dot product (inner product) of vectors
template <typename T>
inline T dot(const Vec3<T>& a, const Vec3<T>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// dot product (inner product) of vectors
inline void normalize(Vec3<FloatT>& a) {
  #if USE_EUCLIDEAN
  FloatT norm2 = dot(a, a);
  a /= sqrt(norm2);
  #else
  a /= a.getNorm1();
  #endif // USE_EUCLIDEAN
}

// Original sampling method
inline void knnSamplingMethod(
  const PointInt& anchor,
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

inline void mySamplingMethod(
  const PointInt& anchor,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& retained,
  std::vector<int32_t>& neighborIndexes,
  int32_t lodIndex,
  int32_t (&localIndexes)[3],
  int64_t (&minDistances)[3])
{
  assert(neighborIndexes.size() <= 64);

  using std::vector;
  
  // Model Parameters
  const int DIST_W = 1;
  // the larger the lod, the longer the distance
  // increase the weight of angle parameter accordingly
  const FloatT ANGLE_W = calAngleW(lodIndex);
  const int K = 3;
  const int MIN_CANDI = K;

  // fall back onto KNN if too few points
  if (neighborIndexes.size() <= MIN_CANDI) {
    knnSamplingMethod(
      anchor, packedVoxel, retained, neighborIndexes, localIndexes,
      minDistances);
    return;
  }

  auto neigSize = neighborIndexes.size();

  vector<FloatT> cost;
  // direction of each point from anchor
  vector<Vec3<FloatT>> dir;
  cost.reserve(neigSize);
  dir.reserve(neigSize);

  for (int k: neighborIndexes) {
    const PointInt &pt = packedVoxel[retained[k]].bposition;
    // initialize cost as distance
    cost.push_back((pt - anchor).getNorm1());

    // translate the point, let the anchor point becomes origin point ...
    dir.push_back(pt - anchor);
    // ... then normalize it
    normalize(dir.back());
  }

  // Use the nearest point as first candidate.
  {
    // find the minimum value
    size_t minI = min_element(cost.begin(), cost.end()) - cost.begin();
    // move it to the front, make sure all all the relative data are swapped.
    swapArrElem(0, minI, cost, dir, neighborIndexes);
  }

  for (size_t i = 1; i < K; ++i) {
    // Update the cost with angular cost
    for (size_t j = i; j < neigSize; ++j) {
      cost[j] += dot(dir[i-1], dir[j]) * ANGLE_W;
    }
    // find the minimum value
    size_t minI = min_element(cost.begin() + i, cost.end()) - cost.begin();
    // move it to the front, make sure all all the relative data are swapped.
    swapArrElem(i, minI, cost, dir, neighborIndexes);
  }

  // write out the result
  std::copy_n(neighborIndexes.begin(), K, localIndexes);
  std::transform(localIndexes, localIndexes+K, minDistances, [&](size_t k){
    const PointInt &pt = packedVoxel[retained[k]].bposition;
    return (anchor - pt).getNorm1();
  });
}

void samplingPoints(
  const PointInt& anchor,
  const std::vector<MortonCodeWithIndex>& packedVoxel,
  const std::vector<uint32_t>& retained,
  std::vector<int32_t>& neighborIndexes,
  int32_t lodIndex,
  int32_t (&localIndexes)[3],
  int64_t (&minDistances)[3]) {
      #if USE_NEW_METHOD
      mySamplingMethod(anchor, packedVoxel, retained, neighborIndexes, lodIndex, localIndexes, minDistances);
      #else
      Kenton::knnSamplingMethod(anchor, packedVoxel, retained, neighborIndexes, localIndexes, minDistances);
      #endif
  }

} // namespace Kenton

}  // namespace pcc