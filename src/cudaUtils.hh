#ifndef CUDAUTILS_HH
#define CUDAUTILS_HH

#if defined(HAVE_HIP)

#include <hip/hip_runtime.h>
__attribute__((unused))
static void checkHip(const hipError_t err, const char *const file, const int line)
{
  if (err == hipSuccess) return;
  fprintf(stderr,"HIP ERROR AT LINE %d OF FILE '%s': %s %s\n",line,file,hipGetErrorName(err),hipGetErrorString(err));
  fflush(stderr);
  exit(err);
}
#define CHECK(X) checkHip(X,__FILE__,__LINE__)

#define launchKernel(kernel,nb,nt,arg1,...) \
        kernel<<<nb,nt>>>(arg1,__VAR_ARGS__)

#define KERNEL_ARGS

#else

#include <cstring>
#include <cstdio>
#include <cassert>
#include <cstdlib>

#define __host__
#define __device__
#define __global__
#define __shared__
#define __launch_bounds__(LB)
#define __syncthreads() do {} while (0)

struct double3 {
    double x,y,z;
};

struct double4 {
    double x,y,z,w;
    //double A,B,C,D;
};

struct int3 {
    int x,y,z;
};

static inline bool
hipGetDeviceCount(int *const c)
{
    *c = 0;
    return true;
}

static inline bool
hipSetDevice(int d)
{
    return true;
}

template <typename T>
bool
hipHostMalloc(T *const *p, size_t size)
{
    *p = static_cast<T*>(malloc(size));
    return *p != nullptr;
}

template <typename T>
bool
hipMalloc(T **const p, size_t size)
{
    *p = static_cast<T*>(malloc(size));
    return *p != nullptr;
}

template <typename T>
bool
hipFree(T *const p)
{
    free(static_cast<void*>(p));
    return true;
}

template <typename T>
bool
hipMemset(T *const p, char v, size_t size)
{
    memset(static_cast<void*>(p), v, size);
    return true;
}

static inline
bool
hipDeviceSynchronize()
{
    return true;
}

using hipEvent_t = bool;

static inline
bool
hipEventDestroy(hipEvent_t e)
{
    return true;
}

static inline
bool
hipEventCreate(hipEvent_t *const e)
{
    *e = true;
    return true;
}

static inline
bool
hipEventRecord(hipEvent_t e)
{
    return true;
}

static inline
bool
hipEventSynchronize(hipEvent_t e)
{
    return true;
}

static inline
bool
hipGetDevice(int *const d)
{
    *d = -1;
    return false;
}

struct hipDeviceProp_t {
    int multiProcessorCount;
};

static inline bool
hipGetDeviceProperties(hipDeviceProp_t *const p, int d)
{
    p->multiProcessorCount = 1;
    return false;
}

template <class T>
inline bool hipOccupancyMaxActiveBlocksPerMultiprocessor(int *const numBlocks, T f, int blockSize, size_t dynSharedMemPerBlk)
{
    *numBlocks = 0;
    return false;
}

#define CHECK(X) do { if (!(X)) { fprintf(stderr,"%s:%d: HIP Call " #X "\n",__FILE__,__LINE__); abort(); } } while (0)

struct dim1 {
    int x;
};

#define KERNEL_ARGS const dim1& gridDim,const dim1& blockIdx,const dim1& blockDim,const dim1& threadIdx,

#define launchKernel(kernel,nb,nt,arg1,...) \
    do { \
        dim1 blockIdx, gridDim={.x = ((nb)*(nt))}; \
        dim1 threadIdx={.x=0}, blockDim={.x = 1}; \
        for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++) { \
            kernel(gridDim,blockIdx,blockDim,threadIdx,arg1,__VA_ARGS__); \
        } \
    } while (0)

#endif

#define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM

template <typename T>
__host__ __device__ T atomicFetchAdd(T *const p, const T x)
{
#ifdef __HIP_DEVICE_COMPILE__
  return atomicAdd(p,x);
#else
  T r = *p;
  *p += x;
  return r;
#endif
}

template <typename T>
void hipCalloc(const size_t n, T *__restrict__ &p)
{
  const size_t bytes = n*sizeof(T);
  CHECK(hipMalloc((void **)&p,bytes));
  CHECK(hipMemset(p,0,bytes));
}

#endif
