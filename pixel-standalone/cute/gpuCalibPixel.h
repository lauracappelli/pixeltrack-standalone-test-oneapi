#ifndef RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
#define RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>
#include <cstdio>

#include "SiPixelGainForHLTonGPU.h"

#include "gpuClusteringConstants.h"

namespace gpuCalibPixel {

  constexpr uint16_t InvId = 9999;  // must be > MaxNumModules

  constexpr float VCaltoElectronGain = 47;         // L2-4: 47 +- 4.7
  constexpr float VCaltoElectronGain_L1 = 50;      // L1:   49.6 +- 2.6
  constexpr float VCaltoElectronOffset = -60;      // L2-4: -60 +- 130
  constexpr float VCaltoElectronOffset_L1 = -670;  // L1:   -670 +- 220

  void calibDigis(cl::sycl::nd_item<1> item_ct1,
                  Input* input,
                  Output* output,
                  uint32_t* __restrict__ moduleStart,        // just to zero first
                  uint32_t* __restrict__ nClustersInModule,  // just to zero them
                  uint32_t* __restrict__ clusModuleStart     // just to zero first
                  ) {
    
    const uint32_t wordCounter = input->wordCounter;
    uint16_t* id = output->moduleInd;
    uint16_t* x = output->xx;
    uint16_t* y = output->yy;
    uint16_t* adc = output->adc;
    int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    // zero for next kernels...
    if (0 == first)
      clusModuleStart[0] = moduleStart[0] = 0;
    for (int i = first; i < gpuClustering::MaxNumModules;
         i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
      nClustersInModule[i] = 0;
    }

    //qua numElements
    for (int i = first; i < numElements; i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
      if (InvId == id[i])
        continue;

      float conversionFactor = id[i] < 96 ? VCaltoElectronGain_L1 : VCaltoElectronGain;
      float offset = id[i] < 96 ? VCaltoElectronOffset_L1 : VCaltoElectronOffset;

      bool isDeadColumn = false, isNoisyColumn = false;

      int row = x[i];
      int col = y[i];
      //auto ret = ped->getPedAndGain(id[i], col, row, isDeadColumn, isNoisyColumn);
      //float pedestal = ret.first;
      //float gain = ret.second;

      float pedestal = 0; float gain = 1.;
      if (isDeadColumn | isNoisyColumn) {
        id[i] = InvId;
        adc[i] = 0;
        printf("bad pixel\n");
      } else {
        float vcal = adc[i] * gain - pedestal * gain;
        adc[i] = sycl::max(100, int(vcal * conversionFactor + offset));
      }
    }
  }
}  // namespace gpuCalibPixel

#endif  // RecoLocalTracker_SiPixelClusterizer_plugins_gpuCalibPixel_h
