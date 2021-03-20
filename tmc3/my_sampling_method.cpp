#include "PCCTMC3Common.h"
#include "test_config.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace pcc {

namespace Kenton {

  // edge case of swapArrElem below
  inline void swapArrElem(size_t i, size_t j)
  {
    // do nothing
  }

  // swap arr[i] & arr[j]. And for each array in arrs, do the same swapping;
  template<typename ArrT, typename... ArrTs>
  inline void swapArrElem(size_t i, size_t j, ArrT& arr, ArrTs&... arrs)
  {
    if (i == j)
      return;
    using std::swap;
    swap(arr[i], arr[j]);
    swapArrElem(i, j, arrs...);
  }

  // dot product (inner product) of vectors
  template<typename T>
  inline T dot(const Vec3<T>& a, const Vec3<T>& b)
  {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  // dot product (inner product) of vectors
  inline void normalize(Vec3<FloatT>& a)
  {
#if USE_EUCLIDEAN
    FloatT norm2 = dot(a, a);
    a /= sqrt(norm2);
#else
    a /= a.getNorm1();
#endif  // USE_EUCLIDEAN
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

  // Model Parameters
  const int DIST_W = 1;
  // the larger the lod, the longer the distance
  // increase the weight of angle parameter accordingly
  const int K = kAttributePredictionMaxNeighbourCount;
  const int MIN_CANDI = K;

  // Reorder the indexes such that k optimal results are at the front of array
  inline void kSurroundingNeighbor(
    std::vector<int32_t>& indexes, std::vector<PointFlt>& dir, int lod)
  {
    const FloatT ANGLE_W = calAngleW(lod);
    std::vector<FloatT> cost;
    cost.reserve(dir.size());

    for (size_t i = 0; i < indexes.size(); ++i) {
      FloatT dist = dir[i].getNorm1();

      // initialize cost as distance
      cost.push_back(dist);

      // Normalize direction vector
      dir[i] /= dist;
    }

    // Use the nearest point as first candidate.
    {
      // find the minimum value
      size_t minI = min_element(cost.begin(), cost.end()) - cost.begin();
      // move it to the front, make sure all all the relative data are swapped.
      swapArrElem(0, minI, cost, dir, indexes);
    }

    for (size_t i = 1; i < K; ++i) {
      // Update the cost with angular cost
      for (size_t j = i; j < indexes.size(); ++j) {
        cost[j] += dot(dir[i - 1], dir[j]) * ANGLE_W;
      }
      // find the minimum value
      size_t minI = min_element(cost.begin() + i, cost.end()) - cost.begin();
      // move it to the front, make sure all all the relative data are swapped.
      swapArrElem(i, minI, cost, dir, indexes);
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

    // fall back onto KNN if too few points
    if (neighborIndexes.size() <= MIN_CANDI) {
      knnSamplingMethod(
        anchor, packedVoxel, retained, neighborIndexes, localIndexes,
        minDistances);
      return;
    }

    // direction of each point from anchor
    vector<Vec3<FloatT>> dir;
    dir.reserve(neighborIndexes.size());

    for (int k : neighborIndexes) {
      const PointInt& pt = packedVoxel[retained[k]].bposition;
      // translate the point, let the anchor point becomes origin point
      dir.push_back(pt - anchor);
    }

    kSurroundingNeighbor(neighborIndexes, dir, lodIndex);

    // write out the result
    std::copy_n(neighborIndexes.begin(), K, localIndexes);
    std::transform(
      localIndexes, localIndexes + K, minDistances, [&](size_t k) {
        const PointInt& pt = packedVoxel[retained[k]].bposition;
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
    int64_t (&minDistances)[3])
  {
#if USE_NEW_METHOD
    mySamplingMethod(
      anchor, packedVoxel, retained, neighborIndexes, lodIndex, localIndexes,
      minDistances);
#else
    knnSamplingMethod(
      anchor, packedVoxel, retained, neighborIndexes, localIndexes,
      minDistances);
#endif
  }

  inline void knnSamplingScalable(
    const AttributeParameterSet& aps,
    const PCCPointSet3& pointCloud,
    const std::vector<MortonCodeWithIndex>& packedVoxel,
    const std::vector<uint32_t>& retained,
    const int32_t nodeSizeLog2,
    const PointInt& anchor,
    const std::vector<int32_t>& neighborIndexes,
    uint32_t& neighborCount,
    PCCNeighborInfo* neighbors)
  {
    for (const auto k : neighborIndexes) {
      updateNearestNeighbor(
        aps, pointCloud, packedVoxel, nodeSizeLog2, retained[k], anchor,
        neighborCount, neighbors);
    }
  }

  inline void mySamplingScalable(
    const AttributeParameterSet& aps,
    const PCCPointSet3& pointCloud,
    const std::vector<MortonCodeWithIndex>& packedVoxel,
    const std::vector<uint32_t>& retained,
    const int32_t nodeSizeLog2,
    const PointInt& anchor,
    std::vector<int32_t>& neighborIndexes,
    uint32_t& neighborCount,
    PCCNeighborInfo* neighbors)
  {
    // fall back onto KNN if too few points
    if (neighborIndexes.size() <= MIN_CANDI) {
      knnSamplingScalable(
        aps, pointCloud, packedVoxel, retained, nodeSizeLog2, anchor,
        neighborIndexes, neighborCount, neighbors);
      return;
    }

    std::vector<PointFlt> dir;
    dir.reserve(neighborIndexes.size());

    for (const auto k : neighborIndexes) {
      const PointInt& pt = packedVoxel[retained[k]].bposition;
      // PointInt point = clacIntermediatePosition(
      //   aps.scalable_lifting_enabled_flag, nodeSizeLog2, pt);
      dir.push_back(pt - anchor);
    }

    kSurroundingNeighbor(neighborIndexes, dir, nodeSizeLog2);

    // write out the result
    std::copy_n(neighborIndexes.begin(), K, neighbors);
    neighborCount = K;
  }

  void samplingPointsScalable(
    const AttributeParameterSet& aps,
    const PCCPointSet3& pointCloud,
    const std::vector<MortonCodeWithIndex>& packedVoxel,
    const std::vector<uint32_t>& retained,
    const int32_t nodeSizeLog2,
    const PointInt& anchor,
    std::vector<int32_t>& neighborIndexes,
    uint32_t& neighborCount,
    PCCNeighborInfo* neighbors)
  {
#if USE_NEW_METHOD
    mySamplingScalable(
      aps, pointCloud, packedVoxel, retained, nodeSizeLog2, anchor,
      neighborIndexes, neighborCount, neighbors);
#else
    knnSamplingScalable(
      aps, pointCloud, packedVoxel, retained, nodeSizeLog2, anchor,
      neighborIndexes, neighborCount, neighbors);
#endif
  }

}  // namespace Kenton

}  // namespace pcc