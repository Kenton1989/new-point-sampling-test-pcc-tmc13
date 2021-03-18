1. for transformType, use default value 0: Hierarchical neighbourhood prediction

- so that buildPredictorsFast in PCCTMC3Common.h will only trigger computeNearestNeighbors, and computeNearestNeighborsScalable don't need to be consider.

2. for lod_neigh_bias, use default value 1,1,1

- So that
  MortonCodeWithIndex::bposition (biased position) == MortonCodeWithIndex::position

3. for predWeightBlending, use default value false

- So that AttributeLods::generate() will not call predictor.blendWeights(). the indexes is not used
