#ifndef digitocluster_oneapi_h_
#define digitocluster_oneapi_h_

#include <CL/sycl.hpp>

class Input;
class Output;

namespace oneapi{

  void digitocluster(const Input* input_d,
                     Output* output_d,
                     const uint32_t wordCounter,
                     bool useQualityInfo,
                     bool includeErrors,
                     bool debug,
#if __SYCL_COMPILER_VERSION <= 20200118
                     // Intel oneAPI beta 4
                     cl::sycl::ordered_queue queue);
#else
                     // Intel SYCL branch
                     cl::sycl::queue queue);
#endif
} //namespace oneapi

#endif  // digitocluster_oneapi_h_


