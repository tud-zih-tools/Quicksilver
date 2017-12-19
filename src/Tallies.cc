#include "Tallies.hh"
#ifdef HAVE_CALIPER
#include<caliper/cali.h>
#endif
#include "utilsMpi.hh"
#include "MC_Time_Info.hh"
#include "MC_Processor_Info.hh"
#include "MonteCarlo.hh"
#include "Globals.hh"
#include "MC_Fast_Timer.hh"

#include <vector>
using std::vector;

void Tallies::CycleInitialize(MonteCarlo* monteCarlo)
{
}

void Tallies::CycleFinalize(MonteCarlo *monteCarlo)
{
#ifdef HAVE_CALIPER
CALI_CXX_MARK_FUNCTION;
#endif
    vector<uint64_t> tal;
    tal.reserve( 13 );
    tal.push_back(_balanceTask[0]._absorb);
    tal.push_back(_balanceTask[0]._census);
    tal.push_back(_balanceTask[0]._escape);
    tal.push_back(_balanceTask[0]._collision);
    tal.push_back(_balanceTask[0]._end);
    tal.push_back(_balanceTask[0]._fission);
    tal.push_back(_balanceTask[0]._produce);
    tal.push_back(_balanceTask[0]._scatter);
    tal.push_back(_balanceTask[0]._start);
    tal.push_back(_balanceTask[0]._source);
    tal.push_back(_balanceTask[0]._rr);
    tal.push_back(_balanceTask[0]._split);
    tal.push_back(_balanceTask[0]._numSegments);
    vector<uint64_t> sum(tal.size());

    mpiAllreduce(&tal[0], &sum[0], tal.size(), MPI_UINT64_T, MPI_SUM, monteCarlo->processor_info->comm_mc_world);

    int index = 0;
    _balanceTask[0]._absorb = sum[index++];
    _balanceTask[0]._census = sum[index++];
    _balanceTask[0]._escape = sum[index++];
    _balanceTask[0]._collision = sum[index++];
    _balanceTask[0]._end = sum[index++];
    _balanceTask[0]._fission = sum[index++];
    _balanceTask[0]._produce = sum[index++];
    _balanceTask[0]._scatter = sum[index++];
    _balanceTask[0]._start = sum[index++];
    _balanceTask[0]._source = sum[index++];
    _balanceTask[0]._rr = sum[index++];
    _balanceTask[0]._split = sum[index++];
    _balanceTask[0]._numSegments = sum[index++];

    PrintSummary(monteCarlo);

    _balanceCumulative.Add(_balanceTask[0]);

    uint64_t newStart = _balanceTask[0]._end;

    for ( auto balanceIter = 0; balanceIter < _balanceTask.size(); balanceIter++)
    {
        _balanceTask[balanceIter].Reset();
    }
    _balanceTask[0]._start = newStart;

    for (int domainIndex = 0; domainIndex < _scalarFluxDomain.size(); domainIndex++)
    {
        if( monteCarlo->_params.simulationParams.coralBenchmark )
            _fluence.compute( domainIndex, _scalarFluxDomain[domainIndex] );

        _cellTallyDomain[domainIndex]._task[0].Reset();
        _scalarFluxDomain[domainIndex]._task[0].Reset();
    }
    _spectrum.UpdateSpectrum(monteCarlo);
}

void Fluence::compute( int domainIndex, ScalarFluxDomain &scalarFluxDomain )
{
#ifdef HAVE_CALIPER
CALI_CXX_MARK_FUNCTION;
#endif
    int numCells = scalarFluxDomain._task[0]._cell.size();

    while( this->_domain.size() <= domainIndex )
    {
        FluenceDomain *newDomain = new FluenceDomain( numCells ); 
        this->_domain.push_back( newDomain );
    }

    FluenceDomain* fluenceDomain = this->_domain[domainIndex];

    for( int cellIndex = 0; cellIndex < numCells; cellIndex++ )
    {
        int numGroups = scalarFluxDomain._task[0]._cell[cellIndex].size();
        for( int groupIndex = 0; groupIndex < numGroups; groupIndex++ )
        {
            fluenceDomain->addCell( cellIndex, scalarFluxDomain._task[0]._cell[cellIndex]._group[groupIndex]);
        }
    }

}

void Tallies::PrintSummary(MonteCarlo *monteCarlo)
{
#ifdef HAVE_CALIPER
CALI_CXX_MARK_FUNCTION;
#endif
   MC_FASTTIMER_STOP(MC_Fast_Timer::cycleFinalize); // stop the finalize timer to get report

   if ( monteCarlo->time_info->cycle == 0 )
   {
       Print0("%-8s ", "cycle");
       _balanceTask[0].PrintHeader();
       Print0("%14s %14s %14s %14s\n", "scalar_flux", "cycleInit", "cycleTracking", "cycleFinalize");
   }

   Print0("%8i ", monteCarlo->time_info->cycle);
   _balanceTask[0].Print();
   double sum = ScalarFluxSum(monteCarlo);
   Print0("%14e %14e %14e %14e\n", sum,
      MC_FASTTIMER_GET_LASTCYCLE(MC_Fast_Timer::cycleInit),
      MC_FASTTIMER_GET_LASTCYCLE(MC_Fast_Timer::cycleTracking),
      MC_FASTTIMER_GET_LASTCYCLE(MC_Fast_Timer::cycleFinalize)
   );

   MC_FASTTIMER_START(MC_Fast_Timer::cycleFinalize); // restart the finalize timer
}

double Tallies::ScalarFluxSum(MonteCarlo *monteCarlo)
{
#ifdef HAVE_CALIPER
CALI_CXX_MARK_FUNCTION;
#endif
    double local_sum = 0.0;

    for (int domainIndex = 0; domainIndex < _scalarFluxDomain.size(); domainIndex++)
    {
      int numCells = _scalarFluxDomain[domainIndex]._task[0]._cell.size();

      for (int cellIndex = 0; cellIndex < numCells; cellIndex++)
      {
        int numGroups = _scalarFluxDomain[domainIndex]._task[0]._cell[cellIndex].size();
        for (int groupIndex = 0; groupIndex < numGroups; groupIndex++)
        {
          local_sum += _scalarFluxDomain[domainIndex]._task[0]._cell[cellIndex]._group[groupIndex];
        }
      }
    }

    double sum = 0.0;
    mpiAllreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, monteCarlo->processor_info->comm_mc_world);

    return sum;
}

void Tallies::InitializeTallies( MonteCarlo *monteCarlo) 
{
#ifdef HAVE_CALIPER
CALI_CXX_MARK_FUNCTION;
#endif
    if( _balanceTask.size() == 0 )
    {
        if( _balanceTask.capacity() == 0 ) 
        {        
            //Reserve replicas number of balance tallies
            _balanceTask.reserve(1,VAR_MEM);
        }

        _balanceTask.Open();
        _balanceTask.push_back( Balance() ); 
        _balanceTask.Close();
    }

    //Initialize the cellTally
    if( _cellTallyDomain.size() == 0 )
    {
        if( _cellTallyDomain.capacity() == 0 ) 
        {   
            _cellTallyDomain.reserve(monteCarlo->domain.size(), VAR_MEM);
        }   
        _cellTallyDomain.Open();
        for (int domainIndex = 0; domainIndex < monteCarlo->domain.size(); domainIndex++)
        {   
            _cellTallyDomain.push_back(CellTallyDomain(&monteCarlo->domain[domainIndex]));
        }   
        _cellTallyDomain.Close();
    }

    //Initialize the scalarFluxTally
    if( _scalarFluxDomain.size() == 0 )
    {
        if( _scalarFluxDomain.capacity() == 0 )
        {
            _scalarFluxDomain.reserve(monteCarlo->domain.size(), VAR_MEM);
        }
        _scalarFluxDomain.Open();
        for (int domainIndex = 0; domainIndex < monteCarlo->domain.size(); domainIndex++)
        {
            _scalarFluxDomain.push_back(ScalarFluxDomain(&monteCarlo->domain[domainIndex],
                                                          monteCarlo->_nuclearData->_energies.size()-1));
        }
        _scalarFluxDomain.Close();
    }
}
