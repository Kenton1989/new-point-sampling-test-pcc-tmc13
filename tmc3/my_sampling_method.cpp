#include "PCCTMC3Common.h"

#include <algorithm>
#include <iterator>
#include <vector>

#include "test_config.h"

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
    a /= sqrt((float)norm2);
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
    std::vector<int32_t>& indexes, std::vector<PointFlt>& points, int32_t lod)
  {
    const FloatT ANGLE_W = calAngleW(lod);
    std::vector<FloatT> cost;
    cost.reserve(points.size());

    size_t firstNon0 = 0;
    for (size_t i = 0; i < indexes.size(); ++i) {
      FloatT dist = points[i].getNorm1();

      // initialize cost as distance
      cost.push_back(dist);

      if (dist > 0) {
        // Normalize direction vector if not 0
        normalize(points[i]);
      } else {
        // record points with 0 distance
        if (firstNon0 != i) {
          swapArrElem(firstNon0, i, cost, points, indexes);
        }
        ++firstNon0;
      }
    }

    // So many 0 distance points, return directly
    if (firstNon0 >= K) {
      return;
    }

    // Use the nearest point as the first candidate.
    {
      // find the minimum value
      size_t minI = min_element(cost.begin(), cost.end()) - cost.begin();
      // move it to the front, make sure all all the relative data are swapped.
      swapArrElem(0, minI, cost, points, indexes);
    }

    for (size_t i = 1; i < K; ++i) {
      // Update the cost with angular cost
      for (size_t j = i; j < indexes.size(); ++j) {
        cost[j] += dot(points[i - 1], points[j]) * ANGLE_W;
      }
      // find the minimum value
      size_t minI = min_element(cost.begin() + i, cost.end()) - cost.begin();
      // move it to the front, make sure all all the relative data are swapped.
      swapArrElem(i, minI, cost, points, indexes);
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

#ifdef DEBUG
    int32_t knnIndexs[3] = {-1, -1, -1};
    int64_t knnDist[3] = {INT64_MAX, INT64_MAX, INT64_MAX};
    if (!calAngleW(lodIndex)) {
      knnSamplingMethod(
        anchor, packedVoxel, retained, neighborIndexes, knnIndexs, knnDist);
    }
#endif  // DEBUG

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

#ifdef DEBUG
    if (!calAngleW(lodIndex)) {
      if (memcmp(knnDist, minDistances, sizeof(knnDist)) != 0) {
        std::cerr << "unexpected" << std::endl;
      }
    }
#endif  // DEBUG
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
      int32_t pointIndex = packedVoxel[retained[k]].index;
      // adjust coordination depending on nodeSize
      PointInt point = clacIntermediatePosition(
        aps.scalable_lifting_enabled_flag, nodeSizeLog2,
        pointCloud[pointIndex]);
      // add the biased direction vector
      dir.push_back(times(point - anchor, aps.lodNeighBias));
    }

    kSurroundingNeighbor(neighborIndexes, dir, nodeSizeLog2);

    int32_t minW = 1 << (nodeSizeLog2 - 1);
    // write out the result
    for (int i = 0; i < K; ++i) {
      neighbors[i].predictorIndex = neighborIndexes[i];

      const PointInt& original =
        pointCloud[packedVoxel[retained[neighborIndexes[i]]].index];
      PointInt point = clacIntermediatePosition(
        aps.scalable_lifting_enabled_flag, nodeSizeLog2, point);

      neighbors[i].weight = times(point - anchor, aps.lodNeighBias).getNorm1();
      if (neighbors[i].weight == 0)
        neighbors[i].weight = minW;
    }
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
#if USE_NEW_SCALABLE_METHOD
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