#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include "SiPixelClustersCUDA.h"

#include "device_unique_ptr.h"
#include "host_unique_ptr.h"
#include "copyAsync.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t maxClusters,
                                         cl::sycl::queue *stream) {
  moduleStart_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters + 1, stream);
  clusInModule_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
  moduleId_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters, stream);
  clusModuleStart_d = cms::cuda::make_device_unique<uint32_t[]>(maxClusters + 1, stream);

  auto view = cms::cuda::make_host_unique<DeviceConstView>(stream);
  view->moduleStart_ = moduleStart_d.get();
  view->clusInModule_ = clusInModule_d.get();
  view->moduleId_ = moduleId_d.get();
  view->clusModuleStart_ = clusModuleStart_d.get();

  view_d = cms::cuda::make_device_unique<DeviceConstView>(stream);
  cms::cuda::copyAsync(view_d, view, stream);
}
