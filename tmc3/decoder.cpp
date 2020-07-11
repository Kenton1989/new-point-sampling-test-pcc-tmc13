/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2017-2018, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "PCCTMC3Decoder.h"

#include <algorithm>
#include <cassert>
#include <string>

#include "PayloadBuffer.h"
#include "PCCPointSet.h"
#include "coordinate_conversion.h"
#include "geometry.h"
#include "geometry_octree.h"
#include "geometry_predictive.h"
#include "hls.h"
#include "io_hls.h"
#include "io_tlv.h"
#include "pcc_chrono.h"
#include "osspecific.h"

namespace pcc {

//============================================================================

PCCTMC3Decoder3::PCCTMC3Decoder3(const DecoderParams& params) : _params(params)
{
  init();
}

//----------------------------------------------------------------------------

void
PCCTMC3Decoder3::init()
{
  _firstSliceInFrame = true;
  _currentFrameIdx = -1;
  _sps = nullptr;
  _gps = nullptr;
  _spss.clear();
  _gpss.clear();
  _apss.clear();

  _ctxtMemOctreeGeom.reset(new GeometryOctreeContexts);
  _ctxtMemPredGeom.reset(new PredGeomContexts);
}

//----------------------------------------------------------------------------

PCCTMC3Decoder3::~PCCTMC3Decoder3() = default;

//============================================================================

static bool
payloadStartsNewSlice(PayloadType type)
{
  return type == PayloadType::kGeometryBrick
    || type == PayloadType::kFrameBoundaryMarker;
}

//============================================================================

int
PCCTMC3Decoder3::decompress(
  const PayloadBuffer* buf, PCCTMC3Decoder3::Callbacks* callback)
{
  // Starting a new geometry brick/slice/tile, transfer any
  // finished points to the output accumulator
  if (!buf || payloadStartsNewSlice(buf->type)) {
    if (size_t numPoints = _currentPointCloud.getPointCount()) {
      for (size_t i = 0; i < numPoints; i++)
        for (int k = 0; k < 3; k++)
          _currentPointCloud[i][k] += _sliceOrigin[k];
      _accumCloud.append(_currentPointCloud);
    }
  }

  if (!buf) {
    // flush decoder, output pending cloud if any
    callback->onOutputCloud(*_sps, _accumCloud);
    _accumCloud.clear();
    return 0;
  }

  switch (buf->type) {
  case PayloadType::kSequenceParameterSet: {
    auto sps = parseSps(*buf);
    convertXyzToStv(&sps);
    storeSps(std::move(sps));
    return 0;
  }

  case PayloadType::kGeometryParameterSet: {
    auto gps = parseGps(*buf);
    // HACK: assume that an SPS has been received prior to the GPS.
    // This is not required, and parsing of the GPS is independent of the SPS.
    // todo(df): move GPS fixup to activation process
    _sps = &_spss.cbegin()->second;
    convertXyzToStv(*_sps, &gps);
    storeGps(std::move(gps));
    return 0;
  }

  case PayloadType::kAttributeParameterSet: {
    auto aps = parseAps(*buf);
    // HACK: assume that an SPS has been received prior to the APS.
    // This is not required, and parsing of the APS is independent of the SPS.
    // todo(df): move APS fixup to activation process
    _sps = &_spss.cbegin()->second;
    convertXyzToStv(*_sps, &aps);
    storeAps(std::move(aps));
    return 0;
  }

  // the frame boundary marker flushes the current frame.
  // NB: frame counter is reset to avoid outputing a runt point cloud
  //     on the next slice.
  case PayloadType::kFrameBoundaryMarker:
    // todo(df): if no sps is activated ...
    callback->onOutputCloud(*_sps, _accumCloud);
    _accumCloud.clear();
    _currentFrameIdx = -1;
    _attrDecoder.reset();
    return 0;

  case PayloadType::kGeometryBrick:
    activateParameterSets(parseGbhIds(*buf));
    if (frameIdxChanged(parseGbh(*_sps, *_gps, *buf, nullptr))) {
      callback->onOutputCloud(*_sps, _accumCloud);
      _accumCloud.clear();
      _firstSliceInFrame = true;
    }

    // avoid accidents with stale attribute decoder on next slice
    _attrDecoder.reset();
    return decodeGeometryBrick(*buf);

  case PayloadType::kAttributeBrick: decodeAttributeBrick(*buf); return 0;

  case PayloadType::kConstantAttribute:
    decodeConstantAttribute(*buf);
    return 0;

  case PayloadType::kTileInventory:
    // NB: the tile inventory is decoded in xyz order.  It may need
    //     conversion if it is used (it currently isn't).
    storeTileInventory(parseTileInventory(*buf));
    return 0;
  }

  // todo(df): error, unhandled payload type
  return 1;
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::storeSps(SequenceParameterSet&& sps)
{
  // todo(df): handle replacement semantics
  _spss.emplace(std::make_pair(sps.sps_seq_parameter_set_id, sps));
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::storeGps(GeometryParameterSet&& gps)
{
  // todo(df): handle replacement semantics
  _gpss.emplace(std::make_pair(gps.gps_geom_parameter_set_id, gps));
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::storeAps(AttributeParameterSet&& aps)
{
  // todo(df): handle replacement semantics
  _apss.emplace(std::make_pair(aps.aps_attr_parameter_set_id, aps));
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::storeTileInventory(TileInventory&& inventory)
{
  // todo(df): handle replacement semantics
  _tileInventory = inventory;
}

//==========================================================================

bool
PCCTMC3Decoder3::frameIdxChanged(const GeometryBrickHeader& gbh) const
{
  // dont treat the first frame in the sequence as a frame boundary
  if (_currentFrameIdx < 0)
    return false;
  return _currentFrameIdx != gbh.frame_idx;
}

//==========================================================================

void
PCCTMC3Decoder3::activateParameterSets(const GeometryBrickHeader& gbh)
{
  // HACK: assume activation of the first SPS and GPS
  // todo(df): parse brick header here for propper sps & gps activation
  //  -- this is currently inconsistent between trisoup and octree
  assert(!_spss.empty());
  assert(!_gpss.empty());
  _sps = &_spss.cbegin()->second;
  _gps = &_gpss.cbegin()->second;
}

//==========================================================================
// Initialise the point cloud storage and decode a single geometry slice.

int
PCCTMC3Decoder3::decodeGeometryBrick(const PayloadBuffer& buf)
{
  assert(buf.type == PayloadType::kGeometryBrick);
  std::cout << "positions bitstream size " << buf.size() << " B\n";

  // todo(df): replace with attribute mapping
  bool hasColour = std::any_of(
    _sps->attributeSets.begin(), _sps->attributeSets.end(),
    [](const AttributeDescription& desc) {
      return desc.attributeLabel == KnownAttributeLabel::kColour;
    });

  bool hasReflectance = std::any_of(
    _sps->attributeSets.begin(), _sps->attributeSets.end(),
    [](const AttributeDescription& desc) {
      return desc.attributeLabel == KnownAttributeLabel::kReflectance;
    });

  _currentPointCloud.clear();
  _currentPointCloud.addRemoveAttributes(hasColour, hasReflectance);

  pcc::chrono::Stopwatch<pcc::chrono::utime_inc_children_clock> clock_user;
  clock_user.start();

  int gbhSize;
  _gbh = parseGbh(*_sps, *_gps, buf, &gbhSize);
  _prevSliceId = _sliceId;
  _sliceId = _gbh.geom_slice_id;
  _sliceOrigin = _gbh.geomBoxOrigin;
  _currentFrameIdx = _gbh.frame_idx;

  // sanity check for loss detection
  if (_gbh.entropy_continuation_flag) {
    assert(!_firstSliceInFrame);
    assert(_gbh.prev_slice_id == _prevSliceId);
  } else {
    // forget (reset) all saved context state at boundary
    if (_sps->entropy_continuation_enabled_flag) {
      _ctxtMemOctreeGeom->reset();
      _ctxtMemPredGeom->reset();
    }
  }

  // set default attribute values (in case an attribute data unit is lost)
  // NB: it is a requirement that geom_num_points_minus1 is correct
  _currentPointCloud.resize(_gbh.footer.geom_num_points_minus1 + 1);
  if (hasColour) {
    auto it = std::find_if(
      _sps->attributeSets.begin(), _sps->attributeSets.end(),
      [](const AttributeDescription& desc) {
        return desc.attributeLabel == KnownAttributeLabel::kColour;
      });
    Vec3<attr_t> defAttrVal =
      Vec3<int>{1 << (it->bitdepth - 1), 1 << (it->bitdepthSecondary - 1),
                1 << (it->bitdepthSecondary - 1)};
    if (!it->attr_default_value.empty())
      for (int k = 0; k < 3; k++)
        defAttrVal[k] = it->attr_default_value[k];
    for (int i = 0; i < _currentPointCloud.getPointCount(); i++)
      _currentPointCloud.setColor(i, defAttrVal);
  }

  if (hasReflectance) {
    auto it = std::find_if(
      _sps->attributeSets.begin(), _sps->attributeSets.end(),
      [](const AttributeDescription& desc) {
        return desc.attributeLabel == KnownAttributeLabel::kReflectance;
      });
    attr_t defAttrVal = 1 << (it->bitdepth - 1);
    if (!it->attr_default_value.empty())
      defAttrVal = it->attr_default_value[0];
    for (int i = 0; i < _currentPointCloud.getPointCount(); i++)
      _currentPointCloud.setReflectance(i, defAttrVal);
  }

  // add a dummy length value to simplify handling the last buffer
  _gbh.geom_stream_len.push_back(buf.size());

  std::vector<std::unique_ptr<EntropyDecoder>> arithmeticDecoders;
  size_t bufRemaining = buf.size() - gbhSize;
  const char* bufPtr = buf.data() + gbhSize;

  for (int i = 0; i <= _gbh.geom_stream_cnt_minus1; i++) {
    arithmeticDecoders.emplace_back(new EntropyDecoder);
    auto& aec = arithmeticDecoders.back();

    // NB: avoid reading beyond the end of the data unit
    int bufLen = std::min(bufRemaining, _gbh.geom_stream_len[i]);

    aec->setBuffer(bufLen, bufPtr);
    aec->enableBypassStream(_sps->cabac_bypass_stream_enabled_flag);
    aec->start();
    bufPtr += bufLen;
    bufRemaining -= bufLen;
  }

  if (_gps->predgeom_enabled_flag)
    decodePredictiveGeometry(
      *_gps, _gbh, _currentPointCloud, *_ctxtMemPredGeom,
      arithmeticDecoders[0].get());
  else if (!_gps->trisoup_enabled_flag) {
    if (!_params.minGeomNodeSizeLog2) {
      decodeGeometryOctree(
        *_gps, _gbh, _currentPointCloud, *_ctxtMemOctreeGeom,
        arithmeticDecoders);
    } else {
      decodeGeometryOctreeScalable(
        *_gps, _gbh, _params.minGeomNodeSizeLog2, _currentPointCloud,
        *_ctxtMemOctreeGeom, arithmeticDecoders);
    }
  } else {
    decodeGeometryTrisoup(
      *_gps, _gbh, _currentPointCloud, *_ctxtMemOctreeGeom,
      arithmeticDecoders);
  }

  // At least the first slice's geometry has been decoded
  _firstSliceInFrame = false;

  clock_user.stop();

  auto total_user =
    std::chrono::duration_cast<std::chrono::milliseconds>(clock_user.count());
  std::cout << "positions processing time (user): "
            << total_user.count() / 1000.0 << " s\n";
  std::cout << std::endl;

  return 0;
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::decodeAttributeBrick(const PayloadBuffer& buf)
{
  assert(buf.type == PayloadType::kAttributeBrick);
  // todo(df): replace assertions with error handling
  assert(_sps);
  assert(_gps);

  // verify that this corresponds to the correct geometry slice
  AttributeBrickHeader abh = parseAbhIds(buf);
  assert(abh.attr_geom_slice_id == _sliceId);

  // todo(df): validate that sps activation is not changed via the APS
  const auto it_attr_aps = _apss.find(abh.attr_attr_parameter_set_id);

  assert(it_attr_aps != _apss.cend());
  const auto& attr_aps = it_attr_aps->second;

  assert(abh.attr_sps_attr_idx < _sps->attributeSets.size());
  const auto& attr_sps = _sps->attributeSets[abh.attr_sps_attr_idx];
  const auto& label = attr_sps.attributeLabel;

  // In order to determinet hat the attribute decoder is reusable, the abh
  // must be inspected.
  // todo(df): remove duplicate parsing from attribute decoder
  abh = parseAbh(*_sps, attr_aps, buf, nullptr);

  pcc::chrono::Stopwatch<pcc::chrono::utime_inc_children_clock> clock_user;

  // replace the attribute decoder if not compatible
  if (!_attrDecoder || !_attrDecoder->isReusable(attr_aps, abh))
    _attrDecoder = makeAttributeDecoder();

  clock_user.start();

  // Convert cartesian positions to spherical for use in attribute coding.
  // NB: this retains the original cartesian positions to restore afterwards
  std::vector<pcc::point_t> altPositions;
  if (attr_aps.spherical_coord_flag) {
    altPositions.resize(_currentPointCloud.getPointCount());

    auto laserOrigin = _gps->geomAngularOrigin - _sliceOrigin;
    auto bboxRpl = convertXyzToRpl(
      laserOrigin, _gps->geom_angular_theta_laser.data(),
      _gps->geom_angular_theta_laser.size(), &_currentPointCloud[0],
      &_currentPointCloud[0] + _currentPointCloud.getPointCount(),
      altPositions.data());

    // todo(df): this needs to be moved to a pre-processing step so that
    // the scale factor can be sent in the APS.
    offsetAndScale(
      bboxRpl.min, abh.attr_coord_conv_scale, altPositions.data(),
      altPositions.data() + altPositions.size());

    _currentPointCloud.swapPoints(altPositions);
  }

  _attrDecoder->decode(
    *_sps, attr_sps, attr_aps, _gbh.footer.geom_num_points_minus1,
    _params.minGeomNodeSizeLog2, buf, _currentPointCloud);

  if (attr_aps.spherical_coord_flag)
    _currentPointCloud.swapPoints(altPositions);

  clock_user.stop();

  std::cout << label << "s bitstream size " << buf.size() << " B\n";

  auto total_user =
    std::chrono::duration_cast<std::chrono::milliseconds>(clock_user.count());
  std::cout << label
            << "s processing time (user): " << total_user.count() / 1000.0
            << " s\n";
  std::cout << std::endl;
}

//--------------------------------------------------------------------------

void
PCCTMC3Decoder3::decodeConstantAttribute(const PayloadBuffer& buf)
{
  assert(buf.type == PayloadType::kConstantAttribute);
  // todo(df): replace assertions with error handling
  assert(_sps);
  assert(_gps);

  ConstantAttributeDataUnit cadu = parseConstantAttribute(*_sps, buf);

  // verify that this corresponds to the correct geometry slice
  assert(cadu.constattr_geom_slice_id == _sliceId);

  assert(cadu.constattr_sps_attr_idx < _sps->attributeSets.size());
  const auto& attrDesc = _sps->attributeSets[cadu.constattr_sps_attr_idx];
  const auto& label = attrDesc.attributeLabel;

  // todo(df): replace with proper attribute mapping
  if (label == KnownAttributeLabel::kColour) {
    Vec3<attr_t> defAttrVal;
    for (int k = 0; k < 3; k++)
      defAttrVal[k] = attrDesc.attr_default_value[k];
    for (int i = 0; i < _currentPointCloud.getPointCount(); i++)
      _currentPointCloud.setColor(i, defAttrVal);
  }

  if (label == KnownAttributeLabel::kReflectance) {
    attr_t defAttrVal = attrDesc.attr_default_value[0];
    for (int i = 0; i < _currentPointCloud.getPointCount(); i++)
      _currentPointCloud.setReflectance(i, defAttrVal);
  }
}

//============================================================================

}  // namespace pcc
