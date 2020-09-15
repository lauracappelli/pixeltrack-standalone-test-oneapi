#ifndef CUDADataFormatsVertexZVertexHeterogeneous_H
#define CUDADataFormatsVertexZVertexHeterogeneous_H

#include "ZVertexSoA.h"
#include "HeterogeneousSoA.h"
#include "PixelTrackHeterogeneous.h"

using ZVertexHeterogeneous = HeterogeneousSoA<ZVertexSoA>;
#ifndef __CUDACC__
#include "../CUDACore/Product.h"
using ZVertexCUDAProduct = cms::cuda::Product<ZVertexHeterogeneous>;
#endif

#endif
