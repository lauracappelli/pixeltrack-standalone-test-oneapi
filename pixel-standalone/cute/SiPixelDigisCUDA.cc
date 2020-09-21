#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SiPixelDigisCUDA.h"

#include "device_unique_ptr.h"
#include "host_unique_ptr.h"
#include "copyAsync.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, cl::sycl::queue *stream) {
  xx_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  yy_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  adc_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d = cms::cuda::make_device_unique<uint16_t[]>(maxFedWords, stream);
  clus_d = cms::cuda::make_device_unique<int32_t[]>(maxFedWords, stream);

  pdigi_d = cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d = cms::cuda::make_device_unique<uint32_t[]>(maxFedWords, stream);

  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = cms::cuda::make_device_unique<DeviceConstView>(stream);
  cms::cuda::copyAsync(view_d, view, stream);
}
/*
cms::cuda::host::unique_ptr<uint16_t[]>
SiPixelDigisCUDA::adcToHostAsync(sycl::queue *stream) const {
  auto ret = cms::cuda::make_host_unique<uint16_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<int32_t[]>
SiPixelDigisCUDA::clusToHostAsync(sycl::queue *stream) const {
  auto ret = cms::cuda::make_host_unique<int32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]>
SiPixelDigisCUDA::pdigiToHostAsync(sycl::queue *stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

cms::cuda::host::unique_ptr<uint32_t[]>
SiPixelDigisCUDA::rawIdArrToHostAsync(sycl::queue *stream) const {
  auto ret = cms::cuda::make_host_unique<uint32_t[]>(nDigis(), stream);
  cms::cuda::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}
*/