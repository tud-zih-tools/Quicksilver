#include "ParticleVaultContainer.hh"
#include "MemoryControl.hh"
#include "qs_assert.hh"

//--------------------------------------------------------------
//------------ParticleVaultContainer Constructor----------------
//Sets up the fixed sized data and pre-allocates the minimum 
//needed for processing and processed vaults
//--------------------------------------------------------------

ParticleVaultContainer::
ParticleVaultContainer( uint64_t vault_size )
: _vaultSize      ( vault_size       )
{
    _processedVault.reserve( vault_size );
    _processingVault.reserve( vault_size );
    _extraVault.reserve( vault_size );
}

//--------------------------------------------------------------
//------------ParticleVaultContainer Destructor-----------------
//Deletes memory allocaetd using the Memory Control class
//--------------------------------------------------------------

ParticleVaultContainer::
~ParticleVaultContainer()
{
}

//--------------------------------------------------------------
//------------swapProcessingProcessedVaults---------------------
//Swaps the vaults from Processed that have particles in them
//with empty vaults from processing to prepair for the next
//cycle
//
//ASSUMPTIONS:: 
//  2) _processingVault is always empty of particles when this is
//      called
//--------------------------------------------------------------

void ParticleVaultContainer::
swapProcessingProcessedVaults()
{
    if (this->_processedVault.size() > 0) {
      std::swap(this->_processingVault, this->_processedVault);
    }
}

//--------------------------------------------------------------
//------------addProcessingParticle-----------------------------
//Adds a particle to the processing particle vault
//--------------------------------------------------------------

void ParticleVaultContainer::
addProcessingParticle( MC_Base_Particle &particle )
{
    _processingVault.pushBaseParticle(particle);
}

//--------------------------------------------------------------
//------------addExtraParticle----------------------------------
//adds a particle to the extra particle vaults (used in kernel)
//--------------------------------------------------------------
void ParticleVaultContainer::
addExtraParticle( MC_Particle &particle)
{
    _extraVault.pushParticle( particle );
}

//--------------------------------------------------------------
//------------cleanExtraVault----------------------------------
//Moves the particles from the _extraVault into the 
//_processedVault
//--------------------------------------------------------------

void ParticleVaultContainer::
cleanExtraVault()
{
  uint64_t size_extra = this->_extraVault.size();
  if( size_extra > 0 )
  {
    const uint64_t fill_size = this->_vaultSize - this->_processingVault.size();
    assert(size_extra < fill_size);
    this->_processingVault.collapse( fill_size, &(this->_extraVault) );
  }
}

