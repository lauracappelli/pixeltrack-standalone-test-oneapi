#ifndef HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h
#define HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdint>

#include "cudaCompat.h"

class AtomicPairCounter {
public:
  using c_type = unsigned long long int;

  AtomicPairCounter() {}
  AtomicPairCounter(c_type i) { counter.ac = i; }

  AtomicPairCounter& operator=(c_type i) {
    counter.ac = i;
    return *this;
  }

  struct Counters {
    uint32_t n;  // in a "One to Many" association is the number of "One"
    uint32_t m;  // in a "One to Many" association is the total number of associations
  };

  union Atomic2 {
    Counters counters;
    c_type ac;
  };

  static constexpr c_type incr = 1UL << 32;

  Counters get() const { return counter.counters; }

  // increment n by 1 and m by i.  return previous value
  __dpct_inline__ Counters add(uint32_t i) {
    c_type c = i;
    c += incr;
    Atomic2 ret;
#ifdef DPCPP_COMPATIBILITY_TEMP
    ret.ac =
        sycl::atomic<AtomicPairCounter::c_type>(sycl::global_ptr<AtomicPairCounter::c_type>(&counter.ac)).fetch_add(c);
#else
    ret.ac = counter.ac;
    counter.ac += c;
#endif
    return ret.counters;
  }

private:
  Atomic2 counter;
};

#endif  // HeterogeneousCore_CUDAUtilities_interface_AtomicPairCounter_h
