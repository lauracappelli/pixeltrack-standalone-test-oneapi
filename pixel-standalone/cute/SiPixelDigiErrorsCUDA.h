#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "PixelErrors.h"
#include "device_unique_ptr.h"
#include "host_unique_ptr.h"
#include "GPUSimpleVector.h"

class SiPixelDigiErrorsCUDA {
public:
  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, sycl::queue* stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  GPU::SimpleVector<PixelErrorCompact>* error() { return error_d.get(); }
  GPU::SimpleVector<PixelErrorCompact> const* error() const { return error_d.get(); }
  GPU::SimpleVector<PixelErrorCompact> const* c_error() const { return error_d.get(); }

  using HostDataError =
      std::pair<GPU::SimpleVector<PixelErrorCompact>, cms::cuda::host::unique_ptr<PixelErrorCompact[]>>;
  HostDataError dataErrorToHostAsync(sycl::queue* stream) const;

  void copyErrorToHostAsync(sycl::queue* stream);

private:
  cms::cuda::device::unique_ptr<PixelErrorCompact[]> data_d;
  cms::cuda::device::unique_ptr<GPU::SimpleVector<PixelErrorCompact>> error_d;
  cms::cuda::host::unique_ptr<GPU::SimpleVector<PixelErrorCompact>> error_h;
  PixelFormatterErrors formatterErrors_h;
};

#endif
