#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <stdlib.h>

#include "prefixScan.h"

using namespace cms::cuda;

template <typename T> struct format_traits {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %d %d\n";
};

template <> struct format_traits<float> {
public:
  static const constexpr char *failed_msg = "failed %d %d %d: %f %f\n";
};

template <typename T>
int SYCL_EXTERNAL testPrefixScan(uint32_t size, sycl::nd_item<3> item_ct1,
                                  sycl::stream stream_ct1, T *ws, T *c, T *co, int dim_subgroup) {

  auto first = item_ct1.get_local_id(2);
  for (auto i = first; i < size; i += item_ct1.get_local_range().get(2))
    c[i] = 1;
  item_ct1.barrier();

  blockPrefixScan(c, co, size, ws, item_ct1, dim_subgroup);
  blockPrefixScan(c, size, ws, item_ct1, dim_subgroup);

  if (!(1 == c[0])) {
    stream_ct1 << "Assertion failed during testPrefixScan (file "
                  "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
    return -1;
  }
  if (!(1 == co[0])) {
    stream_ct1 << "Assertion failed during testPrefixScan (file "
                  "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
    return -1;
  }
  for (auto i = first + 1; i < size; i += item_ct1.get_local_range().get(2)) {
    if (c[i] != c[i - 1] + 1) {
      stream_ct1 << format_traits<unsigned short>::failed_msg << cl::sycl::flush;
      stream_ct1 << format_traits<float>::failed_msg << cl::sycl::flush;
    }
    if (!(c[i] == c[i - 1] + 1)) {
      stream_ct1 << "Assertion failed during testPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
      return -1;
    }
    if (!(c[i] == i + 1)) {
      stream_ct1 << "Assertion failed during testPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
      return -1;
    }
    if (!(c[i] = co[i])) {
      stream_ct1 << "Assertion failed during testPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
      return -1;
    }
  }
  return 0;
}

template <typename T>
int SYCL_EXTERNAL testWarpPrefixScan(uint32_t size, sycl::nd_item<3> item_ct1,
                                      sycl::stream stream_ct1, T *c, T *co, int dim_subgroup) {
  if (!(size <= 32)) {
    stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                 "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
    return -1;
  }

  auto i = item_ct1.get_local_id(2);
  c[i] = 1;
  item_ct1.barrier();

  warpPrefixScan(c, co, i, item_ct1, dim_subgroup);
  warpPrefixScan(c, i, item_ct1, dim_subgroup);
  item_ct1.barrier();

  if (!(1 == c[0])) {
    stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                 "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
    return -1;
  }
  if (!(1 == co[0])) {
    stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                 "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
    return -1;
  }
  if (i != 0) {
    if (c[i] != c[i - 1] + 1)
      stream_ct1 << format_traits<int>::failed_msg << cl::sycl::flush;
    if (!(c[i] == c[i - 1] + 1)) {
      stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
      return -1;
    }
    if (!(c[i] == i + 1)) {
      stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
      return -1;
    }
    if (!(c[i] = co[i])) {
      stream_ct1 << "Assertion failed during testWarpPrefixScan (file "
                   "'prefixScan_t.dp.cpp)\nAborting...\n" << cl::sycl::flush;
      return -1;
    }
  }
  return 0;
}

void init(uint32_t *v, uint32_t val, uint32_t n, sycl::nd_item<3> item_ct1,
          sycl::stream stream_ct1) {
  auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
           item_ct1.get_local_id(2);
  if (i < n)
    v[i] = val;
  if (i == 0)
    stream_ct1 << "init\n";
}

int verify(uint32_t const *v, uint32_t n, sycl::nd_item<3> item_ct1,
            sycl::stream stream_ct1) {
  auto i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
           item_ct1.get_local_id(2);
  if (i < n)
    if (!(v[i] == i + 1)) {
      stream_ct1 << "i = " << i << " v[i] = " << v[i] << " i+1 = " << i+1 << cl::sycl::endl;
      return -1;
    }
  if (i == 0)
    stream_ct1 << "verify\n";
  return 0;
}

int main() {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  constexpr int subgroup_size = 16;

  //max work_item_sizes and max work group size
  std::cout << "\nmax item sizes: ";
  auto max_item_size = dev_ct1.get_info<sycl::info::device::max_work_item_sizes>();
  std::cout << max_item_size[0] << ' ' << max_item_size[1] << ' ' << max_item_size[2];
  int max_item_size_z = max_item_size[2];
  auto max_work_group_size = dev_ct1.get_info<sycl::info::device::max_work_group_size>();
  std::cout << "\nmax work group sizes: " << max_work_group_size << std::endl;
  
  std::cout << "sub-group sizes: ";
  auto dim_subgroup_values = dev_ct1.get_info<sycl::info::device::sub_group_sizes>();
  for (int const &el : dim_subgroup_values) {
    std::cout << el << " ";
  }
  int max_sub_group_size = *std::max_element(std::begin(dim_subgroup_values), std::end(dim_subgroup_values));
  int const dim_subgroup = std::min(16, max_sub_group_size);
  std::cout << "\ndim_subgroup: " << dim_subgroup << std::endl;

  std::cout << "\nwarp level" << std::endl;
  std::cout << "warp 32" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        c_acc_ct1(sycl::range<1>(1024), cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        co_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, dim_subgroup), sycl::range<3>(1, 1, dim_subgroup)),
        [=](sycl::nd_item<3> item_ct1) 
        __attribute__ ((intel_reqd_sub_group_size(subgroup_size)))
        {
          testWarpPrefixScan<int>(32, item_ct1, stream_ct1,
                                  c_acc_ct1.get_pointer(),
                                  co_acc_ct1.get_pointer(),
                                  dim_subgroup);
        });
  });
  dev_ct1.queues_wait_and_throw();

  std::cout << "warp 16" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        c_acc_ct1(sycl::range<1>(1024), cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        co_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, dim_subgroup), sycl::range<3>(1, 1, dim_subgroup)),
        [=](sycl::nd_item<3> item_ct1) 
        __attribute__ ((intel_reqd_sub_group_size(subgroup_size)))
        {
          testWarpPrefixScan<int>(16, item_ct1, stream_ct1,
                                  c_acc_ct1.get_pointer(),
                                  co_acc_ct1.get_pointer(),
                                  dim_subgroup);
        });
  });
  dev_ct1.queues_wait_and_throw();

  std::cout << "warp 5" << std::endl;
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    // accessors to device memory
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        c_acc_ct1(sycl::range<1>(1024), cgh);
    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        co_acc_ct1(sycl::range<1>(1024), cgh);

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, dim_subgroup), sycl::range<3>(1, 1, dim_subgroup)),
        [=](sycl::nd_item<3> item_ct1)
        __attribute__ ((intel_reqd_sub_group_size(subgroup_size)))
        {
          testWarpPrefixScan<int>(5, item_ct1, stream_ct1,
                                  c_acc_ct1.get_pointer(),
                                  co_acc_ct1.get_pointer(),
                                  dim_subgroup);
        });
  });
  dev_ct1.queues_wait_and_throw();

  std::cout << "block level" << std::endl;
  for (int bs = 32; bs <= std::min(max_item_size_z, 1024); bs += 32) {
    //std::cout << "bs " << bs << std::endl;
    for (int j = 1; j <= std::min(max_item_size_z, 1024); ++j) {
      //std::cout << j << std::endl;
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // accessors to device memory
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            ws_acc_ct1(sycl::range<1>(32), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            c_acc_ct1(sycl::range<1>(1024), cgh);
        sycl::accessor<uint16_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            co_acc_ct1(sycl::range<1>(1024), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, bs),
                                           sycl::range<3>(1, 1, bs)),
                         [=](sycl::nd_item<3> item_ct1)
                         __attribute__ ((intel_reqd_sub_group_size(subgroup_size)))
                         {
                           testPrefixScan<uint16_t>(j, item_ct1, stream_ct1,
                                                    ws_acc_ct1.get_pointer(),
                                                    c_acc_ct1.get_pointer(),
                                                    co_acc_ct1.get_pointer(),
                                                    dim_subgroup);
                         });
      });
      dev_ct1.queues_wait_and_throw();

      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        // accessors to device memory
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            ws_acc_ct1(sycl::range<1>(32), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            c_acc_ct1(sycl::range<1>(1024), cgh);
        sycl::accessor<float, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            co_acc_ct1(sycl::range<1>(1024), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, bs),
                                           sycl::range<3>(1, 1, bs)),
                         [=](sycl::nd_item<3> item_ct1)
                         __attribute__ ((intel_reqd_sub_group_size(subgroup_size)))
                         {
                           testPrefixScan<float>(j, item_ct1, stream_ct1,
                                                 ws_acc_ct1.get_pointer(),
                                                 c_acc_ct1.get_pointer(),
                                                 co_acc_ct1.get_pointer(),
                                                 dim_subgroup);
                         });
      });
      dev_ct1.queues_wait_and_throw();
    }
  }
  dev_ct1.queues_wait_and_throw();

  std::cout << "multiblok" << std::endl;
  int num_items = 200;
  auto max_num_items = max_item_size_z * max_work_group_size;
  for (int ksize = 1; ksize < 4; ++ksize) {
    // test multiblock
    // Declare, allocate, and initialize device-accessible pointers for input
    // and output
    num_items *= 10;
    
    if(num_items > max_num_items){
	    printf("Errore. Troppi items. Avvio processo con max_items.\n");
	    num_items = max_num_items;
    }
    uint32_t *d_in;
    uint32_t *d_out1;
    uint32_t *d_out2;

    d_in = (uint32_t *)sycl::malloc_device(num_items * sizeof(uint32_t),
                                           dev_ct1, q_ct1.get_context());
    d_out1 = (uint32_t *)sycl::malloc_device(num_items * sizeof(uint32_t),
                                             dev_ct1, q_ct1.get_context());
    d_out2 = (uint32_t *)sycl::malloc_device(num_items * sizeof(uint32_t),
                                             dev_ct1, q_ct1.get_context());

    auto nthreads = std::min(max_item_size_z, 256);
    auto nblocks = (num_items + nthreads - 1) / nthreads;

    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::stream stream_ct1(64 * 1024, 80, cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                             sycl::range<3>(1, 1, nthreads),
                                         sycl::range<3>(1, 1, nthreads)),
                       [=](sycl::nd_item<3> item_ct1)
                       __attribute__ ((intel_reqd_sub_group_size(subgroup_size)))
                       {
                         init(d_in, 1, num_items, item_ct1, stream_ct1);
                       });
    });

    // the block counter
    int32_t *d_pc;

    d_pc = (int32_t*)sycl::malloc_device(1, dev_ct1, q_ct1.get_context());

   // memset(&d_pc, 0, sizeof(int32_t));

    nthreads = std::min(1024, max_item_size_z);
    nblocks = (num_items + nthreads - 1) / nthreads;
    std::cout << "launch multiBlockPrefixScan " << num_items << ' ' << nblocks
              << std::endl;

    try {
      q_ct1.submit([&](sycl::handler &cgh) {
        // accessors to device memory
        sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(sycl::range<1>(4 * nblocks), cgh);
        sycl::accessor<uint32_t, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            ws_acc_ct1(sycl::range<1>(32), cgh);
        sycl::accessor<bool, 0, sycl::access::mode::read_write,
                       sycl::access::target::local>
            isLastBlockDone_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                               sycl::range<3>(1, 1, nthreads),
                                           sycl::range<3>(1, 1, nthreads)),
                         [=](sycl::nd_item<3> item_ct1)
                         __attribute__ ((intel_reqd_sub_group_size(subgroup_size)))
                         {
                           multiBlockPrefixScan<uint32_t>(
                               d_in, d_out1, num_items, d_pc, item_ct1,
                               dpct_local_acc_ct1.get_pointer(),
                               ws_acc_ct1.get_pointer(),
                               isLastBlockDone_acc_ct1.get_pointer(),
                               dim_subgroup);
			   });
      });
    } catch (std::exception &e) {
      std::cerr << e.what();
    }
    
    try {
      q_ct1.submit([&](sycl::handler &cgh) {
        sycl::stream stream_ct1(64 * 1024, 80, cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nblocks) *
                                               sycl::range<3>(1, 1, nthreads),
                                           sycl::range<3>(1, 1, nthreads)),
                         [=](sycl::nd_item<3> item_ct1)
                         __attribute__ ((intel_reqd_sub_group_size(subgroup_size)))
                         {
                           verify(d_out1, num_items, item_ct1, stream_ct1);
                         });
      });
    } catch (std::exception &e) {
      std::cerr << e.what();
    }

    dev_ct1.queues_wait_and_throw();

  } // ksize
  return 0;
}
