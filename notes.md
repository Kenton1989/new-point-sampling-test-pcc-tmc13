1. do not use lifting transform:

- so that buildPredictorsFast in PCCTMC3Common.h will only trigger computeNearestNeighbors, and computeNearestNeighborsScalable don't need to be consider.
