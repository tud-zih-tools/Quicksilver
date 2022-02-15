#ifndef MC_PROCESSOR_INFO_HH
#define MC_PROCESSOR_INFO_HH

#include "utilsMpi.hh"

class MC_Processor_Info
{
public:

    int rank;
    int num_processors;
    int use_gpu;
    int gpu_id;
    int thread_target;

    MPI_Comm  comm_mc_world;

    MC_Processor_Info()
    : use_gpu(0),
      gpu_id(0),
      thread_target(0),
      comm_mc_world(MPI_COMM_WORLD)
    {
      mpiComm_rank(comm_mc_world, &rank);
      mpiComm_size(comm_mc_world, &num_processors);
    }

};

#endif
