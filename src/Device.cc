#include "Device.hh"

#include "cudaUtils.hh"
#include "MC_Base_Particle.hh"
#include "MonteCarlo.hh"
#include "NuclearData.hh"
#include "ParticleVaultContainer.hh"

void Device::init(MonteCarlo &mc)
{
  assert(domains == nullptr);
  const int domainSize = mc.domain.size();
  CHECK(hipHostMalloc(&domains,domainSize*sizeof(*domains)));

  const int groupSize = mc._nuclearData->_numEnergyGroups;

  int cellSizeSum = 0;
  int nodeSizeSum = 0;
  for (int i = 0; i < domainSize; i++) {
    cellSizeSum += mc.domain[i].cell_state.size();
    nodeSizeSum += mc.domain[i].mesh._node.size();
  }
  
  DeviceCell *cells = nullptr;
  CHECK(hipHostMalloc(&cells,cellSizeSum*sizeof(*cells)));
  double *totals = nullptr;
  CHECK(hipHostMalloc(&totals,cellSizeSum*groupSize*sizeof(*totals)));
  double *groupTallies = nullptr;
  CHECK(hipHostMalloc(&groupTallies,cellSizeSum*groupSize*sizeof(*groupTallies)));
  double3 *nodes = nullptr;
  CHECK(hipHostMalloc(&nodes,nodeSizeSum*sizeof(*nodes)));
  for (int i = 0; i < domainSize; i++) {
    domains[i].cells = cells;
    const int cellSize = mc.domain[i].cell_state.size();
    for (int j = 0; j < cellSize; j++) {
      cells[j] = mc.domain[i].cell_state[j];
      cells[j].totals = totals;
      totals += groupSize;
      cells[j].groupTallies = groupTallies;
      groupTallies += groupSize;
      assert(DeviceCell::numFacets == mc.domain[i].mesh._cellConnectivity[j].num_facets);
      for (int k = 0; k < DeviceCell::numFacets; k++) {
        const MC_General_Plane &plane = mc.domain[i].mesh._cellGeometry[j]._facet[k];
        DeviceFacet &facet = cells[j].facets[k];
        facet.plane = double4{plane.A,plane.B,plane.C,plane.D};
        const int *const p = mc.domain[i].mesh._cellConnectivity[j]._facet[k].point;
        facet.point = int3{p[0],p[1],p[2]};
        const Subfacet_Adjacency &adjacency = mc.domain[i].mesh._cellConnectivity[j]._facet[k].subfacet;
        facet.adjacentCell = adjacency.adjacent.cell;
        facet.adjacentDomain = adjacency.adjacent.domain;
        facet.adjacentFacet = adjacency.adjacent.facet;
        facet.event = adjacency.event;
      }
      for (int k = 0; k < DeviceCell::numQuadPoints; k++) {
        cells[j].quadPoints[k] = mc.domain[i].mesh._cellConnectivity[j]._point[k];
      }
    }
    cells += cellSize;
    domains[i].nodes = nodes;
    const int nodeSize = mc.domain[i].mesh._node.size();
    for (int j = 0; j < nodeSize; j++) {
      const MC_Vector &node = mc.domain[i].mesh._node[j];
      nodes[j] = double3{node.x,node.y,node.z};
    }
    nodes += nodeSize;
  }

  assert(mats == nullptr);
  const int matSize = mc._materialDatabase->_mat.size();
  CHECK(hipHostMalloc(&mats,matSize*sizeof(*mats)));
  
  int isoSizeSum = 0;
  for (int i = 0; i < matSize; i++) isoSizeSum += mc._materialDatabase->_mat[i]._iso.size();

  DeviceIsotope *isos = nullptr;
  CHECK(hipHostMalloc(&isos,isoSizeSum*sizeof(*isos)));
  for (int i = 0; i < matSize; i++) {
    mats[i].isos = isos;
    const int isoSize = mc._materialDatabase->_mat[i]._iso.size();
    mats[i].isoSize = isoSize;
    for (int j = 0; j < isoSize; j++) isos[j] = mc._materialDatabase->_mat[i]._iso[j];
    isos += isoSize;
    mats[i].mass = mc._materialDatabase->_mat[i]._mass;
  }

  const int ndiSize = mc._nuclearData->_isotopes.size();
  CHECK(hipHostMalloc(&isotopes,ndiSize*sizeof(*isotopes)));
  reactionSize = mc._nuclearData->_isotopes[0]._species[0]._reactions.size();
  const int rSizeP1 = reactionSize+1;
  assert(groupSize == mc._nuclearData->_isotopes[0]._species[0]._reactions[0]._crossSection.size());
  for (const auto &isotope : mc._nuclearData->_isotopes) {
    for (const auto &species : isotope._species) {
      assert(rSizeP1 == species._reactions.size()+1);
      for (const auto &reaction: species._reactions) {
        assert(groupSize == reaction._crossSection.size());
      }
    }
  }

  DeviceReaction *rs = nullptr;
  CHECK(hipHostMalloc(&rs,ndiSize*rSizeP1*sizeof(*rs)));
  double *xs = nullptr;
  CHECK(hipHostMalloc(&xs,ndiSize*rSizeP1*groupSize*sizeof(*xs)));
  for (int i = 0; i < ndiSize; i++) {
    isotopes[i].reactions = rs;
    for (int j = 0; j < rSizeP1; j++) {
      isotopes[i].reactions[j].crossSections = xs;
      xs += groupSize;
    }
    rs += rSizeP1;
  }

  nuBar = mc._nuclearData->_isotopes[0]._species[0]._reactions[0]._nuBar;
  for (int i = 0; i < ndiSize; i++) {
    for (int j = 1; j < rSizeP1; j++) {
      isotopes[i].reactions[j].type = mc._nuclearData->_isotopes[i]._species[0]._reactions[j-1]._reactionType;
      assert(nuBar == mc._nuclearData->_isotopes[i]._species[0]._reactions[j-1]._nuBar);
    }
    for (int k = 0; k < groupSize; k++) {
      double sum = 0;
      for (int j = 1; j < rSizeP1; j++) {
        const double xs = mc._nuclearData->_isotopes[i]._species[0]._reactions[j-1]._crossSection[k];
        sum += xs;
        isotopes[i].reactions[j].crossSections[k] = xs;
      }
      isotopes[i].reactions[0].crossSections[k] = sum;
    }
  }

  CHECK(hipHostMalloc(&particleSizes,PARTICLE_SIZES_SIZE*sizeof(*particleSizes)));
  memset(particleSizes,0,PARTICLE_SIZES_SIZE*sizeof(*particleSizes));

  CHECK(hipHostMalloc(&tallies,TALLIES_SIZE*sizeof(*tallies)));

  {
    const long bytes = sizeof(*processing)*mc._particleVaultContainer->getVaultSize();
    assert(bytes);
    CHECK(hipHostMalloc(&processing,bytes));
    memset(processing,0,bytes);
    CHECK(hipHostMalloc(&processed,bytes));
    memset(processed,0,bytes);
    CHECK(hipHostMalloc(&extras,bytes));
    memset(extras,0,bytes);
  }

  {
    logLow = log(mc._nuclearData->_energies[0]);
    numGroups = mc._nuclearData->_numEnergyGroups;
    const double delta = (log(mc._nuclearData->_energies[numGroups])-logLow)/double(numGroups);
    divDelta = 1.0/delta;
  }
}

void Device::cycleInit(MonteCarlo &mc)
{
  const int groupSize = mc._nuclearData->_numEnergyGroups;
  const int domainSize = mc.domain.size();
  int cellSizeSum = 0;
  for (int i = 0; i < domainSize; i++) cellSizeSum += mc.domain[i].cell_state.size();
  const int bytes = cellSizeSum*groupSize*sizeof(double);
  memset(domains->cells->totals,0,bytes);
  memset(domains->cells->groupTallies,0,bytes);
  memset(tallies,0,TALLIES_SIZE*sizeof(*tallies));
}
  
void Device::cycleFinalize(MonteCarlo &mc)
{
  const int groupSize = mc._nuclearData->_numEnergyGroups;
  const int domainSize = mc.domain.size();
  for (int i = 0; i < domainSize; i++) {
    const int cellSize = mc.domain[i].cell_state.size();
    for (int j = 0; j < cellSize; j++) {
      for (int k = 0; k < groupSize; k++) {
        mc._tallies->_scalarFluxDomain[i]._task[0]._cell[j]._group[k] = domains[i].cells[j].groupTallies[k];
      }
    }
  }
  mc._tallies->_balanceTask[0]._numSegments = tallies[Tallies::SEGMENTS];
  mc._tallies->_balanceTask[0]._collision = tallies[Tallies::COLLISION];
  mc._tallies->_balanceTask[0]._scatter = tallies[Tallies::SCATTER];
  mc._tallies->_balanceTask[0]._absorb = tallies[Tallies::ABSORB];
  mc._tallies->_balanceTask[0]._fission = tallies[Tallies::FISSION];
  mc._tallies->_balanceTask[0]._produce = tallies[Tallies::PRODUCE];
}

DeviceParticle &DeviceParticle::operator=(const MC_Base_Particle &that)
{
  identifier = that.identifier;
  species = that.species;
  return *this;
}

DeviceParticle &DeviceParticle::operator=(const MC_Particle &that)
{
  return *this = MC_Base_Particle(that);
}

bool DeviceParticle::operator==(const MC_Base_Particle &that) const
{
  return (
      (identifier == that.identifier) &&
      (species == that.species)
      );
}

