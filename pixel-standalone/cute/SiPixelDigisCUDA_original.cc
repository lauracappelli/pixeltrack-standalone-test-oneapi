//#include "CUDADataFormats/SiPixelDigisCUDA.h"

//#include "CUDACore/device_unique_ptr.h"
//#include "CUDACore/host_unique_ptr.h"
//#include "CUDACore/copyAsync.h"

#include<CL/sycl.hpp>
#include<dpct/dpct.hpp>

SiPixelDigisCUDA::SiPixelDigisCUDA(unsigned long maxFedWords, cl::sycl::queue *stream1) {
  xx_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream1);
  yy_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream1);
  adc_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream1);
  moduleInd_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream1);
  clus_d = cms::cuda::make_device_unique<int32_t[]>(maxFedWords, stream1);

  pdigi_d = cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream1);
  rawIdArr_d = cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream1);

  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream1);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = cms::cuda::make_device_unique<DeviceConstView>(stream1);
  cms::cuda::copyAsync(view_d, view, stream1);
}
/*
cms::cuda::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(cudaStream_t stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}*/
