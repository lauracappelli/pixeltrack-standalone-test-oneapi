/* Sushil Dubey, Shashi Dugad, TIFR, July 2017
 *
 * File Name: RawToClusterGPU.cu
 * Description: It converts Raw data into Digi Format on GPU
 * Finaly the Output of RawToDigi data is given to pixelClusterizer
 *
**/

// C++ includes
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

// CUDA includes

// cub includes

// CMSSW includes
/*#include "../CUDADataFormats/gpuClusteringConstants.h"
#include "../CUDACore/cudaCheck.h"
#include "../CUDACore/device_unique_ptr.h"
#include "../CUDACore/host_unique_ptr.h"
*/
#include "../CondFormats/SiPixelFedCablingMapGPU.h"

#include "cute/gpuCalibPixel.h"
#include "cute/gpuClusterChargeCut.h"
#include "cute/gpuClustering.h"

// local includes
#include "SiPixelRawToClusterGPUKernel.h"

namespace pixelgpudetails {

  // number of words for all the FEDs
  constexpr uint32_t MAX_FED_WORDS = pixelgpudetails::MAX_FED * pixelgpudetails::MAX_WORD;

  SiPixelRawToClusterGPUKernel::WordFedAppender::WordFedAppender() {
    word_ = cms::cuda::make_host_noncached_unique<unsigned int[]>(MAX_FED_WORDS, cudaHostAllocWriteCombined);
    fedId_ = cms::cuda::make_host_noncached_unique<unsigned char[]>(MAX_FED_WORDS, cudaHostAllocWriteCombined);
  }

  void SiPixelRawToClusterGPUKernel::WordFedAppender::initializeWordFed(int fedId,
                                                                        unsigned int wordCounterGPU,
                                                                        const uint32_t *src,
                                                                        unsigned int length) {
    std::memcpy(word_.get() + wordCounterGPU, src, sizeof(uint32_t) * length);
    std::memset(fedId_.get() + wordCounterGPU / 2, fedId - 1200, length / 2);
  }

  ////////////////////

  uint32_t getLink(uint32_t ww) {
    return ((ww >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask);
  }

  uint32_t getRoc(uint32_t ww) { return ((ww >> pixelgpudetails::ROC_shift) & pixelgpudetails::ROC_mask); }

  uint32_t getADC(uint32_t ww) { return ((ww >> pixelgpudetails::ADC_shift) & pixelgpudetails::ADC_mask); }

  bool isBarrel(uint32_t rawId) { return (1 == ((rawId >> 25) & 0x7)); }

  pixelgpudetails::DetIdGPU getRawId(const SiPixelFedCablingMapGPU *cablingMap,
                                                uint8_t fed,
                                                uint32_t link,
                                                uint32_t roc) {
    uint32_t index = fed * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
    pixelgpudetails::DetIdGPU detId = {
        cablingMap->RawId[index], cablingMap->rocInDet[index], cablingMap->moduleId[index]};
    return detId;
  }

  //reference http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/dd/d31/FrameConversion_8cc_source.html
  //http://cmslxr.fnal.gov/source/CondFormats/SiPixelObjects/src/PixelROC.cc?v=CMSSW_9_2_0#0071
  // Convert local pixel to pixelgpudetails::global pixel
  pixelgpudetails::Pixel frameConversion(
      bool bpix, int side, uint32_t layer, uint32_t rocIdInDetUnit, pixelgpudetails::Pixel local) {
    int slopeRow = 0, slopeCol = 0;
    int rowOffset = 0, colOffset = 0;

    if (bpix) {
      if (side == -1 && layer != 1) {  // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }       // if roc
      } else {  // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
        if (rocIdInDetUnit < 8) {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = rocIdInDetUnit * pixelgpudetails::numColsInRoc;
        } else {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (16 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        }
      }

    } else {             // fpix
      if (side == -1) {  // pannel 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }
      } else {  // pannel 2
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8 - rocIdInDetUnit) * pixelgpudetails::numColsInRoc - 1;
        } else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2 * pixelgpudetails::numRowsInRoc - 1;
          colOffset = (rocIdInDetUnit - 8) * pixelgpudetails::numColsInRoc;
        }

      }  // side
    }

    uint32_t gRow = rowOffset + slopeRow * local.row;
    uint32_t gCol = colOffset + slopeCol * local.col;
    //printf("Inside frameConversion row: %u, column: %u\n", gRow, gCol);
    pixelgpudetails::Pixel global = {gRow, gCol};
    return global;
  }

  uint8_t conversionError(uint8_t fedId, uint8_t status, sycl::stream stream_ct1, bool debug = false) {
    uint8_t errorType = 0;

    // debug = true;

    switch (status) {
      case (1): {
        if (debug)
          stream_ct1 << "Error in Fed: %i, invalid channel Id (errorType = 35\n)" << cl::sycl::endl;
        errorType = 35;
        break;
      }
      case (2): {
        if (debug)
          stream_ct1 << "Error in Fed: %i, invalid ROC Id (errorType = 36)\n" << cl::sycl::endl;
        errorType = 36;
        break;
      }
      case (3): {
        if (debug)

          stream_ct1 << "Error in Fed: %i, invalid dcol/pixel value (errorType = 37)\n" << cl::sycl::endl;
        errorType = 37;
        break;
      }
      case (4): {
        if (debug)
          stream_ct1 << "Error in Fed: %i, dcol/pixel read out of order (errorType = 38)\n" << cl::sycl::endl;
        errorType = 38;
        break;
      }
      default:
        if (debug)
          stream_ct1 << "Cabling check returned unexpected result, status = %i\n" << cl::sycl::endl;
    };

    return errorType;
  }

  bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol) {
    uint32_t numRowsInRoc = 80;
    uint32_t numColsInRoc = 52;

    /// row and collumn in ROC representation
    return ((rocRow < numRowsInRoc) & (rocCol < numColsInRoc));
  }

  bool dcolIsValid(uint32_t dcol, uint32_t pxid) { return ((dcol < 26) & (2 <= pxid) & (pxid < 162)); }

  uint8_t checkROC(
      uint32_t errorWord, uint8_t fedId, uint32_t link, const SiPixelFedCablingMapGPU *cablingMap,
      sycl::stream stream_ct1, bool debug = false) {
    uint8_t errorType = (errorWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ERROR_mask;
    if (errorType < 25)
      return 0;
    bool errorFound = false;

    switch (errorType) {
      case (25): {
        errorFound = true;
        uint32_t index = fedId * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + 1;
        if (index > 1 && index <= cablingMap->size) {
          if (!(link == cablingMap->link[index] && 1 == cablingMap->roc[index]))
            errorFound = false;
        }
        if (debug and errorFound)
          stream_ct1 << "Invalid ROC = 25 found (errorType = 25)\n" << cl::sycl::endl;
        break;
      }
      case (26): {
        if (debug)
          stream_ct1 << "Gap word found (errorType = 26)\n" << cl::sycl::endl;
        errorFound = true;
        break;
      }
      case (27): {
        if (debug)
          stream_ct1 << "Dummy word found (errorType = 27)\n" << cl::sycl::endl;
        errorFound = true;
        break;
      }
      case (28): {
        if (debug)
          stream_ct1 << "Error fifo nearly full (errorType = 28)\n" << cl::sycl::endl;
        errorFound = true;
        break;
      }
      case (29): {
        if (debug)
          stream_ct1 << "Timeout on a channel (errorType = 29)\n" << cl::sycl::endl;
        if ((errorWord >> pixelgpudetails::OMIT_ERR_shift) & pixelgpudetails::OMIT_ERR_mask) {
          if (debug)
            stream_ct1 << "...first errorType=29 error, this gets masked out\n" << cl::sycl::endl;
        }
        errorFound = true;
        break;
      }
      case (30): {
        if (debug)
          stream_ct1 << "TBM error trailer (errorType = 30)\n" cl::sycl::endl;
        int StateMatch_bits = 4;
        int StateMatch_shift = 8;
        uint32_t StateMatch_mask = ~(~uint32_t(0) << StateMatch_bits);
        int StateMatch = (errorWord >> StateMatch_shift) & StateMatch_mask;
        if (StateMatch != 1 && StateMatch != 8) {
          if (debug)
            stream_ct1 << "FED error 30 with unexpected State Bits (errorType = 30)\n" << cl::sycl::endl;
        }
        if (StateMatch == 1)
          errorType = 40;  // 1=Overflow -> 40, 8=number of ROCs -> 30
        errorFound = true;
        break;
      }
      case (31): {
        if (debug)
          stream_ct1 << "Event number error (errorType = 31)\n" << cl::sycl::endl;
        errorFound = true;
        break;
      }
      default:
        errorFound = false;
    };

    return errorFound ? errorType : 0;
  }

  uint32_t getErrRawID(uint8_t fedId,
                                  uint32_t errWord,
                                  uint32_t errorType,
                                  const SiPixelFedCablingMapGPU *cablingMap,
                                  bool debug = false) {
    uint32_t rID = 0xffffffff;

    switch (errorType) {
      case 25:
      case 30:
      case 31:
      case 36:
      case 40: {
        //set dummy values for cabling just to get detId from link
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = 1;
        uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      case 29: {
        int chanNmbr = 0;
        const int DB0_shift = 0;
        const int DB1_shift = DB0_shift + 1;
        const int DB2_shift = DB1_shift + 1;
        const int DB3_shift = DB2_shift + 1;
        const int DB4_shift = DB3_shift + 1;
        const uint32_t DataBit_mask = ~(~uint32_t(0) << 1);

        int CH1 = (errWord >> DB0_shift) & DataBit_mask;
        int CH2 = (errWord >> DB1_shift) & DataBit_mask;
        int CH3 = (errWord >> DB2_shift) & DataBit_mask;
        int CH4 = (errWord >> DB3_shift) & DataBit_mask;
        int CH5 = (errWord >> DB4_shift) & DataBit_mask;
        int BLOCK_bits = 3;
        int BLOCK_shift = 8;
        uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
        int BLOCK = (errWord >> BLOCK_shift) & BLOCK_mask;
        int localCH = 1 * CH1 + 2 * CH2 + 3 * CH3 + 4 * CH4 + 5 * CH5;
        if (BLOCK % 2 == 0)
          chanNmbr = (BLOCK / 2) * 9 + localCH;
        else
          chanNmbr = ((BLOCK - 1) / 2) * 9 + 4 + localCH;
        if ((chanNmbr < 1) || (chanNmbr > 36))
          break;  // signifies unexpected result

        // set dummy values for cabling just to get detId from link if in Barrel
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = 1;
        uint32_t link = chanNmbr;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      case 37:
      case 38: {
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        uint32_t roc = (errWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ROC_mask;
        uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
        uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999)
          rID = rID_temp;
        break;
      }
      default:
        break;
    };

    return rID;
  }

  // Kernel to perform Raw to Digi conversion
  void RawToDigi_kernel(const SiPixelFedCablingMapGPU *cablingMap,
                                   const unsigned char *modToUnp,
                                   const uint32_t wordCounter,
                                   const uint32_t *word,
                                   const uint8_t *fedIds,
                                   uint16_t *xx,
                                   uint16_t *yy,
                                   uint16_t *adc,
                                   uint32_t *pdigi,
                                   uint32_t *rawIdArr,
                                   uint16_t *moduleId,
                                   GPU::SimpleVector<PixelErrorCompact> *err,
                                   bool useQualityInfo,
                                   bool includeErrors,
                                   bool debug,
                                   sycl::nd_item<3> item_ct1,
                                   sycl::stream stream_ct1) {
    //if (threadIdx.x==0) printf("Event: %u blockIdx.x: %u start: %u end: %u\n", eventno, blockIdx.x, begin, end);

    int32_t first = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
    for (int32_t iloop = first, nend = wordCounter; iloop < nend;
         iloop += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
      auto gIndex = iloop;
      xx[gIndex] = 0;
      yy[gIndex] = 0;
      adc[gIndex] = 0;
      bool skipROC = false;

      uint8_t fedId = fedIds[gIndex / 2];  // +1200;

      // initialize (too many coninue below)
      pdigi[gIndex] = 0;
      rawIdArr[gIndex] = 0;
      moduleId[gIndex] = 9999;

      uint32_t ww = word[gIndex];  // Array containing 32 bit raw data
      if (ww == 0) {
        // 0 is an indicator of a noise/dead channel, skip these pixels during clusterization
        continue;
      }

      uint32_t link = getLink(ww);  // Extract link
      uint32_t roc = getRoc(ww);    // Extract Roc in link
      pixelgpudetails::DetIdGPU detId = getRawId(cablingMap, fedId, link, roc);

      uint8_t errorType = checkROC(ww, fedId, link, cablingMap, stream_ct1, debug);
      skipROC = (roc < pixelgpudetails::maxROCIndex) ? false : (errorType != 0);
      if (includeErrors and skipROC) {
        uint32_t rID = getErrRawID(fedId, ww, errorType, cablingMap, debug);
        err->push_back(PixelErrorCompact{rID, ww, errorType, fedId});
        continue;
      }

      uint32_t rawId = detId.RawId;
      uint32_t rocIdInDetUnit = detId.rocInDet;
      bool barrel = isBarrel(rawId);

      uint32_t index = fedId * MAX_LINK * MAX_ROC + (link - 1) * MAX_ROC + roc;
      if (useQualityInfo) {
        skipROC = cablingMap->badRocs[index];
        if (skipROC)
          continue;
      }
      skipROC = modToUnp[index];
      if (skipROC)
        continue;

      uint32_t layer = 0;                   //, ladder =0;
      int side = 0, panel = 0, module = 0;  //disk = 0, blade = 0

      if (barrel) {
        layer = (rawId >> pixelgpudetails::layerStartBit) & pixelgpudetails::layerMask;
        module = (rawId >> pixelgpudetails::moduleStartBit) & pixelgpudetails::moduleMask;
        side = (module < 5) ? -1 : 1;
      } else {
        // endcap ids
        layer = 0;
        panel = (rawId >> pixelgpudetails::panelStartBit) & pixelgpudetails::panelMask;
        //disk  = (rawId >> diskStartBit_) & diskMask_;
        side = (panel == 1) ? -1 : 1;
        //blade = (rawId >> bladeStartBit_) & bladeMask_;
      }

      // ***special case of layer to 1 be handled here
      pixelgpudetails::Pixel localPix;
      if (layer == 1) {
        uint32_t col = (ww >> pixelgpudetails::COL_shift) & pixelgpudetails::COL_mask;
        uint32_t row = (ww >> pixelgpudetails::ROW_shift) & pixelgpudetails::ROW_mask;
        localPix.row = row;
        localPix.col = col;
        if (includeErrors) {
          if (not rocRowColIsValid(row, col)) {
            uint8_t error = conversionError(fedId, 3, stream_ct1, debug);  //use the device function and fill the arrays
            err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
            if (debug)
              /*
              DPCT1015:33: Output needs adjustment.
              */
              stream_ct1 << "BPIX1  Error status: %i\n" << cl::sycl::endl;
            continue;
          }
        }
      } else {
        // ***conversion rules for dcol and pxid
        uint32_t dcol = (ww >> pixelgpudetails::DCOL_shift) & pixelgpudetails::DCOL_mask;
        uint32_t pxid = (ww >> pixelgpudetails::PXID_shift) & pixelgpudetails::PXID_mask;
        uint32_t row = pixelgpudetails::numRowsInRoc - pxid / 2;
        uint32_t col = dcol * 2 + pxid % 2;
        localPix.row = row;
        localPix.col = col;
        if (includeErrors and not dcolIsValid(dcol, pxid)) {
          uint8_t error = conversionError(fedId, 3, stream_ct1, debug);
          err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
          if (debug)
            stream_ct1 << "Error status: %i %d %d %d %d\n" << cl::sycl::endl;
          continue;
        }
      }

      pixelgpudetails::Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
      xx[gIndex] = globalPix.row;  // origin shifting by 1 0-159
      yy[gIndex] = globalPix.col;  // origin shifting by 1 0-415
      adc[gIndex] = getADC(ww);
      pdigi[gIndex] = pixelgpudetails::pack(globalPix.row, globalPix.col, adc[gIndex]);
      moduleId[gIndex] = detId.moduleId;
      rawIdArr[gIndex] = rawId;
    }  // end of loop (gIndex < end)

  }  // end of Raw to Digi kernel

  void fillHitsModuleStart(uint32_t const *__restrict__ cluStart, uint32_t *__restrict__ moduleStart,
                           sycl::nd_item<3> item_ct1, uint32_t *ws) {
    assert(gpuClustering::MaxNumModules < 2048);  // easy to extend at least till 32*1024
    assert(1 == gridDim.x);
    assert(0 == blockIdx.x);

    int first = item_ct1.get_local_id(2);

    // limit to MaxHitsInModule;
    for (int i = first, iend = gpuClustering::MaxNumModules; i < iend; i += item_ct1.get_local_range().get(2)) {
      moduleStart[i + 1] = sycl::min(gpuClustering::maxHitsInModule(), (const unsigned int)(cluStart[i]));
    }

    blockPrefixScan(moduleStart + 1, moduleStart + 1, 1024, item_ct1, ws);
    blockPrefixScan(moduleStart + 1025, moduleStart + 1025, gpuClustering::MaxNumModules - 1024, item_ct1, ws);

    for (int i = first + 1025, iend = gpuClustering::MaxNumModules + 1; i < iend;
         i += item_ct1.get_local_range().get(2)) {
      moduleStart[i] += moduleStart[1024];
    }
    item_ct1.barrier();

#ifdef GPU_DEBUG
    assert(0 == moduleStart[0]);
    auto c0 = std::min(gpuClustering::maxHitsInModule(), cluStart[0]);
    assert(c0 == moduleStart[1]);
    assert(moduleStart[1024] >= moduleStart[1023]);
    assert(moduleStart[1025] >= moduleStart[1024]);
    assert(moduleStart[gpuClustering::MaxNumModules] >= moduleStart[1025]);

    for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += blockDim.x) {
      if (0 != i)
        assert(moduleStart[i] >= moduleStart[i - i]);
      // [BPX1, BPX2, BPX3, BPX4,  FP1,  FP2,  FP3,  FN1,  FN2,  FN3, LAST_VALID]
      // [   0,   96,  320,  672, 1184, 1296, 1408, 1520, 1632, 1744,       1856]
      if (i == 96 || i == 1184 || i == 1744 || i == gpuClustering::MaxNumModules)
        printf("moduleStart %d %d\n", i, moduleStart[i]);
    }
#endif

    // avoid overflow
    constexpr auto MAX_HITS = gpuClustering::MaxNumClusters;
    for (int i = first, iend = gpuClustering::MaxNumModules + 1; i < iend; i += item_ct1.get_local_range().get(2)) {
      if (moduleStart[i] > MAX_HITS)
        moduleStart[i] = MAX_HITS;
    }
  }

  // Interface to outside
  void SiPixelRawToClusterGPUKernel::makeClustersAsync(const SiPixelFedCablingMapGPU *cablingMap,
                                                       const unsigned char *modToUnp,
                                                       const SiPixelGainForHLTonGPU *gains,
                                                       const WordFedAppender &wordFed,
                                                       PixelFormatterErrors &&errors,
                                                       const uint32_t wordCounter,
                                                       const uint32_t fedCounter,
                                                       bool useQualityInfo,
                                                       bool includeErrors,
                                                       bool debug,
                                                       sycl::queue *stream) {
    nDigis = wordCounter;

#ifdef GPU_DEBUG
    std::cout << "decoding " << wordCounter << " digis. Max is " << pixelgpudetails::MAX_FED_WORDS << std::endl;
#endif

    digis_d = SiPixelDigisCUDA(pixelgpudetails::MAX_FED_WORDS, stream);
    if (includeErrors) {
      digiErrors_d = SiPixelDigiErrorsCUDA(pixelgpudetails::MAX_FED_WORDS, std::move(errors), stream);
    }
    clusters_d = SiPixelClustersCUDA(gpuClustering::MaxNumModules, stream);

    nModules_Clusters_h = cms::cuda::make_host_unique<uint32_t[]>(2, stream);

    if (wordCounter)  // protect in case of empty event....
    {
      const int threadsPerBlock = 512;
      const int blocks = (wordCounter + threadsPerBlock - 1) / threadsPerBlock;  // fill it all

      assert(0 == wordCounter % 2);
      // wordCounter is the total no of words in each event to be trasfered on device
      auto word_d = cms::cuda::make_device_unique<uint32_t[]>(wordCounter, stream);
      auto fedId_d = cms::cuda::make_device_unique<uint8_t[]>(wordCounter, stream);

      /*
      DPCT1003:35: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      stream->memcpy(word_d.get(), wordFed.word(), wordCounter * sizeof(uint32_t))
      /*
      DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      stream->memcpy(fedId_d.get(), wordFed.fedId(), wordCounter * sizeof(uint8_t) / 2)

      // Launch rawToDigi kernel
      stream->submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // helper variables defined
        auto word_d_get_ct3 = word_d.get();
        auto fedId_d_get_ct4 = fedId_d.get();
        auto digis_d_xx_ct5 = digis_d.xx();
        auto digis_d_yy_ct6 = digis_d.yy();
        auto digis_d_adc_ct7 = digis_d.adc();
        auto digis_d_pdigi_ct8 = digis_d.pdigi();
        auto digis_d_rawIdArr_ct9 = digis_d.rawIdArr();
        auto digis_d_moduleInd_ct10 = digis_d.moduleInd();
        auto digiErrors_d_error_ct11 = digiErrors_d.error();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock),
                                           sycl::range<3>(1, 1, threadsPerBlock)),
                         [=](sycl::nd_item<3> item_ct1) {
                           RawToDigi_kernel(cablingMap,
                                            modToUnp,
                                            wordCounter,
                                            word_d_get_ct3,
                                            fedId_d_get_ct4,
                                            digis_d_xx_ct5,
                                            digis_d_yy_ct6,
                                            digis_d_adc_ct7,
                                            digis_d_pdigi_ct8,
                                            digis_d_rawIdArr_ct9,
                                            digis_d_moduleInd_ct10,
                                            digiErrors_d_error_ct11,
                                            useQualityInfo,
                                            includeErrors,
                                            debug,
                                            item_ct1,
                                            stream_ct1);
                         });
      });
      /*
      DPCT1010:37: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);
#ifdef GPU_DEBUG
      item_ct1.barrier();
      //cudaDeviceSynchronize();			!! DA MODIFICARE
      //cudaCheck(cudaGetLastError());
#endif

      if (includeErrors) {
        digiErrors_d.copyErrorToHostAsync(stream);
      }
    }
    // End of Raw2Digi and passing data for clustering

    {
      // clusterizer ...
      using namespace gpuClustering;
      int threadsPerBlock = 256;
      int blocks =
          (std::max(int(wordCounter), int(gpuClustering::MaxNumModules)) + threadsPerBlock - 1) / threadsPerBlock;

      stream->submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // helper variables defined
        auto digis_d_moduleInd_ct0 = digis_d.moduleInd();
        auto digis_d_c_xx_ct1 = digis_d.c_xx();
        auto digis_d_c_yy_ct2 = digis_d.c_yy();
        auto digis_d_adc_ct3 = digis_d.adc();
        auto clusters_d_moduleStart_ct6 = clusters_d.moduleStart();
        auto clusters_d_clusInModule_ct7 = clusters_d.clusInModule();
        auto clusters_d_clusModuleStart_ct8 = clusters_d.clusModuleStart();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock),
                                           sycl::range<3>(1, 1, threadsPerBlock)),
                         [=](sycl::nd_item<3> item_ct1) {
                           calibDigis(digis_d_moduleInd_ct0,
                                      digis_d_c_xx_ct1,
                                      digis_d_c_yy_ct2,
                                      digis_d_adc_ct3,
                                      gains,
                                      wordCounter,
                                      clusters_d_moduleStart_ct6,
                                      clusters_d_clusInModule_ct7,
                                      clusters_d_clusModuleStart_ct8,
                                      item_ct1,
                                      stream_ct1);
                         });
      });
      /*
      DPCT1010:38: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);
#ifdef GPU_DEBUG
      item_ct1.barrier();
      //cudaDeviceSynchronize();      		!! DA MODIFICARE
      //cudaCheck(cudaGetLastError());
#endif

#ifdef GPU_DEBUG
      std::cout << "CUDA countModules kernel launch with " << blocks << " blocks of " << threadsPerBlock
                << " threads\n" << std::endl;
#endif

      stream->submit([&](sycl::handler &cgh) {
        // helper variables defined
        auto digis_d_c_moduleInd_ct0 = digis_d.c_moduleInd();
        auto clusters_d_moduleStart_ct1 = clusters_d.moduleStart();
        auto digis_d_clus_ct2 = digis_d.clus();

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock),
                              sycl::range<3>(1, 1, threadsPerBlock)),
            [=](sycl::nd_item<3> item_ct1) {
              countModules(
                  digis_d_c_moduleInd_ct0, clusters_d_moduleStart_ct1, digis_d_clus_ct2, wordCounter, item_ct1);
            });
      });
      /*
      DPCT1010:39: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);

      // read the number of modules into a data member, used by getProduct())
      /*
      DPCT1003:40: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      stream->memcpy(&(nModules_Clusters_h[0]), clusters_d.moduleStart(), sizeof(uint32_t))
	}

      threadsPerBlock = 256;
      blocks = MaxNumModules;
#ifdef GPU_DEBUG
      std::cout << "CUDA findClus kernel launch with " << blocks << " blocks of " << threadsPerBlock << " threads" << std::endl;
#endif
      stream->submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // accessors to device memory
        sycl::accessor<int, 0, sycl::access::mode::read_write, sycl::access::target::local> msize_acc_ct1(cgh);
        sycl::accessor<Hist, 0, sycl::access::mode::read_write, sycl::access::target::local> hist_acc_ct1(cgh);
        sycl::accessor<typename Hist::Counter, 1, sycl::access::mode::read_write, sycl::access::target::local>
            ws_acc_ct1(sycl::range<1>(32), cgh);
        sycl::accessor<unsigned int, 0, sycl::access::mode::read_write, sycl::access::target::local>
            foundClusters_acc_ct1(cgh);

        // helper variables defined
        auto digis_d_c_moduleInd_ct0 = digis_d.c_moduleInd();
        auto digis_d_c_xx_ct1 = digis_d.c_xx();
        auto digis_d_c_yy_ct2 = digis_d.c_yy();
        auto clusters_d_c_moduleStart_ct3 = clusters_d.c_moduleStart();
        auto clusters_d_clusInModule_ct4 = clusters_d.clusInModule();
        auto clusters_d_moduleId_ct5 = clusters_d.moduleId();
        auto digis_d_clus_ct6 = digis_d.clus();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock),
                                           sycl::range<3>(1, 1, threadsPerBlock)),
                         [=](sycl::nd_item<3> item_ct1) {
                           findClus(digis_d_c_moduleInd_ct0,
                                    digis_d_c_xx_ct1,
                                    digis_d_c_yy_ct2,
                                    clusters_d_c_moduleStart_ct3,
                                    clusters_d_clusInModule_ct4,
                                    clusters_d_moduleId_ct5,
                                    digis_d_clus_ct6,
                                    wordCounter,
                                    item_ct1,
                                    stream_ct1,
                                    msize_acc_ct1.get_pointer(),
                                    hist_acc_ct1.get_pointer(),
                                    ws_acc_ct1.get_pointer(),
                                    foundClusters_acc_ct1.get_pointer());
                         });
      });
      /*
      DPCT1010:41: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);
#ifdef GPU_DEBUG
      item_ct1.barrier();
      //cudaDeviceSynchronize();
      //cudaCheck(cudaGetLastError());
#endif

      // apply charge cut
      stream->submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // accessors to device memory
        sycl::accessor<int32_t, 1, sycl::access::mode::read_write, sycl::access::target::local> charge_acc_ct1(
            sycl::range<1>(1024 /*MaxNumClustersPerModules*/), cgh);
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write, sycl::access::target::local> ok_acc_ct1(
            sycl::range<1>(1024 /*MaxNumClustersPerModules*/), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::access::target::local> newclusId_acc_ct1(
            sycl::range<1>(1024 /*MaxNumClustersPerModules*/), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write, sycl::access::target::local> ws_acc_ct1(
            sycl::range<1>(32), cgh);

        // helper variables defined
        auto digis_d_moduleInd_ct0 = digis_d.moduleInd();
        auto digis_d_c_adc_ct1 = digis_d.c_adc();
        auto clusters_d_c_moduleStart_ct2 = clusters_d.c_moduleStart();
        auto clusters_d_clusInModule_ct3 = clusters_d.clusInModule();
        auto clusters_d_c_moduleId_ct4 = clusters_d.c_moduleId();
        auto digis_d_clus_ct5 = digis_d.clus();

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) * sycl::range<3>(1, 1, threadsPerBlock),
                                           sycl::range<3>(1, 1, threadsPerBlock)),
                         [=](sycl::nd_item<3> item_ct1) {
                           clusterChargeCut(digis_d_moduleInd_ct0,
                                            digis_d_c_adc_ct1,
                                            clusters_d_c_moduleStart_ct2,
                                            clusters_d_clusInModule_ct3,
                                            clusters_d_c_moduleId_ct4,
                                            digis_d_clus_ct5,
                                            wordCounter,
                                            item_ct1,
                                            stream_ct1,
                                            charge_acc_ct1.get_pointer(),
                                            ok_acc_ct1.get_pointer(),
                                            newclusId_acc_ct1.get_pointer(),
                                            ws_acc_ct1.get_pointer());
                         });
      });
      /*
      DPCT1010:42: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
      */
      //cudaCheck(0);

      // count the module start indices already here (instead of
      // rechits) so that the number of clusters/hits can be made
      // available in the rechit producer without additional points of
      // synchronization/ExternalWork

      // MUST be ONE block
      stream->submit([&](sycl::handler &cgh) {
        // accessors to device memory
        sycl::accessor<uint32_t, 1, sycl::access::mode::read_write, sycl::access::target::local> ws_acc_ct1(
            sycl::range<1>(32), cgh);

        // helper variables defined
        auto clusters_d_c_clusInModule_ct0 = clusters_d.c_clusInModule();
        auto clusters_d_clusModuleStart_ct1 = clusters_d.clusModuleStart();

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, 1024), sycl::range<3>(1, 1, 1024)), [=](sycl::nd_item<3> item_ct1) {
              fillHitsModuleStart(
                  clusters_d_c_clusInModule_ct0, clusters_d_clusModuleStart_ct1, item_ct1, ws_acc_ct1.get_pointer());
            });
      });

      // last element holds the number of all clusters
      /*
      DPCT1003:43: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
      */
      if((stream->memcpy(
              &(nModules_Clusters_h[1]), clusters_d.clusModuleStart() + gpuClustering::MaxNumModules, sizeof(uint32_t))!=0){
		      std::cout << "Errore" << std::endl;      
	}

#ifdef GPU_DEBUG
      item_ct1.barrier();
      //cudaDeviceSynchronize();
      //cudaCheck(cudaGetLastError());
#endif

    }  // end clusterizer scope
  }
}  // namespace pixelgpudetails
