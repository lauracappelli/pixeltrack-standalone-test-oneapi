#ifndef modules_h_
#define modules_h_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

namespace gpuClustering {
constexpr uint32_t MaxNumModules = 2000;
constexpr uint16_t InvId = 9999; // must be > MaxNumModules
} // namespace gpuClustering

inline int countModules(const uint16_t *id, int size) {
  int modules = 0;
  for (int i = 0; i < size; ++i) {
    if (id[i] == gpuClustering::InvId)
      continue;
    auto j = i - 1;
    while (j >= 0 and id[j] == gpuClustering::InvId) {
      --j;
    }
    if (j < 0 or id[j] != id[i]) {
      ++modules;
    }
  }
  return modules;
}

void countModules(uint16_t const *__restrict__ id,
                  uint32_t *__restrict__ moduleStart,
                  int32_t *__restrict__ clusterId, int numElements,
                  sycl::nd_item<3> item_ct1) {
  int first = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
  for (int i = first; i < numElements;
       i += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
    clusterId[i] = i;
    if (InvId == id[i])
      continue;
    auto j = i - 1;
    while (j >= 0 and id[j] == InvId)
      --j;
    if (j < 0 or id[j] != id[i]) {
      // boundary...
      auto loc = dpct::atomic_fetch_compare_inc(moduleStart, MaxNumModules);
      moduleStart[loc + 1] = i;
    }
  }
}

#endif // modules_h_
