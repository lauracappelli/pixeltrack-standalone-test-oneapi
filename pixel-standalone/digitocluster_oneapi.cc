#include <algorithm>
#include <cmath>
#include <cstdio>

#include <CL/sycl.hpp>

#include "GPUSimpleVector.h"
#include "input.h"
#include "modules.h"
#include "output.h"
#include "digitocluster_oneapi.h"
#include "kernel/gpuCalibPixel.h"

namespace oneapi {

  class calibDigis_kernel_;

  void digitocluster(const Input* input_d,
                     Output* output_d,
                     const uint32_t wordCounter,
                     bool useQualityInfo,
                     bool includeErrors,
                     bool debug,
#if __SYCL_COMPILER_VERSION <= 20200118
                     // Intel oneAPI beta 4
                     cl::sycl::ordered_queue queue) try {
#else
                     // Intel SYCL branch
                     cl::sycl::queue queue) try {
#endif

    const uint32_t blockSize = std::min({queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>(),
                                         queue.get_device().get_info<cl::sycl::info::device::max_work_item_sizes>()[0],
                                         4096ul});  // upper limit from trial and error
    const uint32_t blocks = std::min((wordCounter + blockSize - 1) / blockSize,
                                     queue.get_device().get_info<cl::sycl::info::device::max_compute_units>());
    const uint32_t threads = blocks * blockSize;

    if (debug) {
      std::cout << "work groups: " << blocks << ", work items per group: " << blockSize << std::endl;
    }

    //eseguo il kernel calibDigis
    auto clusters_d = SiPixelClustersCUDA(gpuClustering::MaxNumModules, stream);
    auto clusters_d_moduleStart = clusters_d.moduleStart();
    auto clusters_d_clusInModule = clusters_d.clusInModule();
    auto clusters_d_clusModuleStart = clusters_d.clusModuleStart();
    queue.submit([&](cl::sycl::handler& cgh) {
      cgh.parallel_for<calibDigis_kernel_>(
          cl::sycl::nd_range<1>{{threads}, {blockSize}}, [=](cl::sycl::nd_item<1> item) {
            calibDigis_kernel(item, input_d, output_d, gains,
                              clusters_d_moduleStart,
                              clusters_d_clusInModule,
                              clusters_d_clusModuleStart);
          });
    });
    queue.wait_and_throw();
    if (debug) {
      queue.submit([&](cl::sycl::handler& cgh) {
        cgh.single_task([=]() { count_modules_kernel(input_d, output_d); });
      });
    }
  } catch (cl::sycl::exception const& exc) {
    std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
    std::exit(1);
  } //fine digitocluster

} //namespace oneapi