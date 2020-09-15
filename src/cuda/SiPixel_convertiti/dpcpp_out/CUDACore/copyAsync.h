#ifndef HeterogeneousCore_CUDAUtilities_copyAsync_h
#define HeterogeneousCore_CUDAUtilities_copyAsync_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "cudaCheck.h"
#include "device_unique_ptr.h"
#include "host_unique_ptr.h"

#include <type_traits>

namespace cms {
  namespace cuda {
    // Single element
template <typename T>
inline void copyAsync(device::unique_ptr<T> &dst,
                      const host::unique_ptr<T> &src, sycl::queue *stream) {
      // Shouldn't compile for array types because of sizeof(T), but
      // let's add an assert with a more helpful message
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(dst.get(), src.get(), sizeof(T)), 0));
    }

template <typename T>
inline void copyAsync(host::unique_ptr<T> &dst,
                      const device::unique_ptr<T> &src, sycl::queue *stream) {
      static_assert(std::is_array<T>::value == false,
                    "For array types, use the other overload with the size parameter");
  /*
  DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(dst.get(), src.get(), sizeof(T)), 0));
    }

    // Multiple elements
template <typename T>
inline void copyAsync(device::unique_ptr<T[]> &dst,
                      const host::unique_ptr<T[]> &src, size_t nelements,
                      sycl::queue *stream) {
  /*
  DPCT1003:10: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(dst.get(), src.get(), nelements * sizeof(T)), 0));
    }

template <typename T>
inline void copyAsync(host::unique_ptr<T[]> &dst,
                      const device::unique_ptr<T[]> &src, size_t nelements,
                      sycl::queue *stream) {
  /*
  DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  cudaCheck((stream->memcpy(dst.get(), src.get(), nelements * sizeof(T)), 0));
    }
  }  // namespace cuda
}  // namespace cms

#endif
