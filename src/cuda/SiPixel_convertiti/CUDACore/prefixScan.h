#ifndef HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
#define HeterogeneousCore_CUDAUtilities_interface_prefixScan_h

#include <CL/sycl.hpp>
#include <cstdint>
#include <dpct/dpct.hpp>
#include <math.h>
#include <tbb/tbb_exception.h>

#ifdef DPCPP_COMPATIBILITY_TEMP

template <typename T>
void __dpct_inline__ SYCL_EXTERNAL warpPrefixScan(T const *__restrict__ ci,
                                                  T *__restrict__ co,
                                                  uint32_t i,
                                                  sycl::nd_item<3> item_ct1,
                                                  int dim_subgroup // 16 o 8
                                                  ) {
  // ci and co may be the same
  auto x = ci[i];
  auto laneId = item_ct1.get_local_id(2) % dim_subgroup;
#pragma unroll
  for (int offset = 1; offset < dim_subgroup; offset <<= 1) {
    /*
    DPCT1023:4: The DPC++ sub-group does not support mask options for
     * shuffle_up.
    */
    auto y = item_ct1.get_sub_group().shuffle_up(x, offset);
    if (laneId >= offset)
      x += y;
  }
  co[i] = x;
}

template <typename T>
void __dpct_inline__ SYCL_EXTERNAL warpPrefixScan(T *c, uint32_t i,
                                                  sycl::nd_item<3> item_ct1, int dim_subgroup) {
  auto x = c[i];
  auto laneId = item_ct1.get_local_id(2) % dim_subgroup;
#pragma unroll
  for (int offset = 1; offset < dim_subgroup; offset <<= 1) {
    /*
    DPCT1023:5: The DPC++ sub-group does not support mask options for
     * shuffle_up.
    */
    auto y = item_ct1.get_sub_group().shuffle_up(x, offset);
    if (laneId >= offset)
      x += y;
  }
  c[i] = x;
}

#endif

namespace cms {
namespace cuda {

// limited to 32*32 elements....
template <typename VT, typename T>
__dpct_inline__ SYCL_EXTERNAL int blockPrefixScan(VT const *ci, VT *co,
                                                   uint32_t size, T *ws,
                                                   sycl::nd_item<3> item_ct1
#ifndef DPCPP_COMPATIBILITY_TEMP
                                                   = nullptr
#endif
                                                  , int dim_subgroup // 8 o 16
) {
#ifdef DPCPP_COMPATIBILITY_TEMP
  if (!(ws)) { // aggiungere il messaggio di errore!
    return -1;
  }
  if (!(size <= 1024)) { // aggiungere il messaggio di errore!
    return -1;
  }
  if (!(0 == item_ct1.get_local_range().get(2) % dim_subgroup)) { // aggiungere il messaggio di errore!
    return -1;
  }
  auto first = item_ct1.get_local_id(2);

  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2)) {

    warpPrefixScan(ci, co, i, item_ct1, dim_subgroup);
    auto laneId = item_ct1.get_local_id(2) % dim_subgroup;
    auto warpId = i / dim_subgroup;
    if (!(warpId < dim_subgroup)) { // aggiungere il messaggio di errore!
      return -1;
    }
    if ((dim_subgroup - 1) == laneId)
      ws[warpId] = co[i];
  }
  item_ct1.barrier();
  if (size <= dim_subgroup)
    return 0;
  if (item_ct1.get_local_id(2) < dim_subgroup)
    warpPrefixScan(ws, item_ct1.get_local_id(2), item_ct1, dim_subgroup);
  item_ct1.barrier();
  for (auto i = first + dim_subgroup; i < size; i += item_ct1.get_local_range().get(2)) {
    auto warpId = i / dim_subgroup;
    co[i] += ws[warpId - 1];
  }
  item_ct1.barrier();
#else
  co[0] = ci[0];
  for (uint32_t i = 1; i < size; ++i)
    co[i] = ci[i] + co[i - 1];
#endif
  return 0;
}

// same as above, may remove
// limited to 32*32 elements....
template <typename T>
__dpct_inline__ SYCL_EXTERNAL int blockPrefixScan(T *c, uint32_t size, T *ws,
                                                   sycl::nd_item<3> item_ct1
#ifndef DPCPP_COMPATIBILITY_TEMP
                                                   = nullptr
#endif
                                                  , int dim_subgroup
) {
#ifdef DPCPP_COMPATIBILITY_TEMP
  if (!(ws)) { // aggiungere il messaggio di errore!
    return -1;
  }
  if (!(size <= 1024)) { // aggiungere il messaggio di errore!
    return -1;
  }
  if (!(0 == item_ct1.get_local_range().get(2) % dim_subgroup)) { // aggiungere il messaggio di errore!
    return -1;
  }
  auto first = item_ct1.get_local_id(2);

  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2)) {
    warpPrefixScan(c, i, item_ct1, dim_subgroup);
    auto laneId = item_ct1.get_local_id(2) % dim_subgroup;
    auto warpId = i / dim_subgroup;
    if (!(warpId < dim_subgroup)) { // aggiungere il messaggio di errore!
      return -1;
    }
    if ((dim_subgroup - 1) == laneId)
      ws[warpId] = c[i];
  }
  item_ct1.barrier();
  if (size <= dim_subgroup)
    return 0;
  if (item_ct1.get_local_id(2) < dim_subgroup)
    warpPrefixScan(ws, item_ct1.get_local_id(2), item_ct1, dim_subgroup);
  item_ct1.barrier();
  for (auto i = first + dim_subgroup; i < size; i += item_ct1.get_local_range().get(2)) {
    auto warpId = i / dim_subgroup;
    c[i] += ws[warpId - 1];
  }
  item_ct1.barrier();
#else
  for (uint32_t i = 1; i < size; ++i)
    c[i] += c[i - 1];
#endif
  return 0;
}

#ifdef DPCPP_COMPATIBILITY_TEMP
// see
// https://stackoverflow.com/questions/40021086/can-i-obtain-the-amount-of-allocated-dynamic-shared-memory-from-within-a-kernel/40021087#40021087
/*__dpct_inline__ unsigned dynamic_smem_size() {
  unsigned ret;
  asm volatile("mov.u32 %0, %dynamic_smem_size;" : "=r"(ret));
  return ret;
}*/
#endif

// in principle not limited....
template <typename T>
int multiBlockPrefixScan(T *const ici, T *ico, int32_t size, int32_t *pc,
                          sycl::nd_item<3> item_ct1, uint8_t *dpct_local, T *ws,
                          bool *isLastBlockDone, int dim_subgroup) {
  volatile T const *ci = ici;
  volatile T *co = ico;

#ifdef DPCPP_COMPATIBILITY_TEMP
  /*if (!(sizeof(T) * item_ct1.get_group_range().get(2) <= dynamic_smem_size())) { // aggiungere il messaggio di errore!
    abort();
  }
  assert(sizeof(T) * item_ct1.get_group_range().get(2) <=
  dynamic_smem_size()); // size of psum below*/
#endif
  if (!(item_ct1.get_local_range().get(2) * item_ct1.get_group_range().get(2) >=
        size)) { // aggiungere il messaggio di errore!
    return -1;
  }
  // first each block does a scan
  int off = item_ct1.get_local_range().get(2) * item_ct1.get_group(2);
  if (size - off > 0)
    blockPrefixScan(
        ci + off, co + off,
        sycl::min(int(item_ct1.get_local_range(2)), (int)(size - off)), ws,
        item_ct1, dim_subgroup);

  // count blocks that finished

  if (0 == item_ct1.get_local_id(2)) {
    //__threadfence();
    //item_ct1.barrier();
    auto value = dpct::atomic_fetch_add(pc, 1); // block counter
    *isLastBlockDone = (value == (int(item_ct1.get_group_range(2)) - 1));
  }

  item_ct1.barrier();

  if (!(*isLastBlockDone))
    return 0;

  if (!(int(item_ct1.get_group_range().get(2)) == *pc)) { // aggiungere il messaggio di errore!
    return -1;
  }

  // good each block has done its work and now we are left in last block

  // let's get the partial sums from each block
  auto psum = (T *)dpct_local;
  for (int i = item_ct1.get_local_id(2), ni = item_ct1.get_group_range(2);
       i < ni; i += item_ct1.get_local_range().get(2)) {
    auto j = item_ct1.get_local_range().get(2) * i +
             item_ct1.get_local_range().get(2) - 1;
    psum[i] = (j < size) ? co[j] : T(0);
  }
  item_ct1.barrier();
  blockPrefixScan(psum, psum, item_ct1.get_group_range(2), ws, item_ct1, dim_subgroup);

  // now it would have been handy to have the other blocks around...
  for (int i = item_ct1.get_local_id(2) + item_ct1.get_local_range().get(2),
           k = 0;
       i < size; i += item_ct1.get_local_range().get(2), ++k) {
    co[i] += psum[k];
  }
  return 0;
}
} // namespace cuda
} // namespace cms

#endif // HeterogeneousCore_CUDAUtilities_interface_prefixScan_h
