#include "cudaFunctions.hh"
#ifdef HAVE_CALIPER
#include<caliper/cali.h>
#endif
#include "cudaUtils.hh"
#ifdef HAVE_HIP
__global__ static void warmup(const bool stop)
{
  if (stop) abort();
}
void warmup_kernel()
{
  warmup<<<1,1>>>(false);
  CHECK(hipDeviceSynchronize());
}
#endif
