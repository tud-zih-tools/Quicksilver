#ifndef MC_BASE_PARTICLE
#define MC_BASE_PARTICLE

#include "portability.hh"

#include "MC_Vector.hh"
#include "MC_RNG_State.hh"
#include "MC_Particle.hh"
#include "MC_Location.hh"
#include "DirectionCosine.hh"
#include "MC_Tally_Event.hh"


struct MC_Data_Member_Operation
{
    public:
    enum Enum
    {
        Count   = 0,
        Pack    = 1,
        Unpack  = 2,
        Reset   = 3
    };
};


class MC_Base_Particle
{
  public:

    static void Cycle_Setup();
    static void Update_Counts();

    MC_Base_Particle();
    explicit MC_Base_Particle(  const MC_Particle &particle);
    MC_Base_Particle(  const MC_Base_Particle &particle);

    MC_Base_Particle&  operator= ( const MC_Particle& );

    int particle_id_number() const;

    // serialize the vault
    void Serialize(int *int_data, double *float_data, char *char_data,
                  int &int_index, int &float_index, int &char_index,
                  MC_Data_Member_Operation::Enum mode);

    // return a location
    MC_Location Get_Location() const;

    // copy contents to a string
    void Copy_Particle_Base_To_String(std::string &output_string) const;

    inline double Get_Energy() const                     { return kinetic_energy; }

    MC_Vector                          coordinate;
    MC_Vector                          velocity;
    double                             kinetic_energy;
    double                             weight;
    double                             time_to_census;
    double                             age;
    double                             num_mean_free_paths;
    double                             num_segments;

    uint64_t                           random_number_seed;
    uint64_t                           identifier;

    MC_Tally_Event::Enum               last_event;
    int                                num_collisions;
    int                                breed;
    int                                domain;
    int                                cell;

    static int                         num_base_ints;   // Number of ints for communication
    static int                         num_base_floats; // Number of floats for communication
    static int                         num_base_chars;  // Number of chars for communication

  private:
    
};

//----------------------------------------------------------------------------------------------------------------------
//  Return a MC_Location given domain, cell, facet.
//----------------------------------------------------------------------------------------------------------------------

inline MC_Location MC_Base_Particle::Get_Location() const
{
    return MC_Location(domain, cell, 0);

}  // End Get_Location


//----------------------------------------------------------------------------------------------------------------------
//  Base information for a particle.
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
// Default constructor.
//----------------------------------------------------------------------------------------------------------------------
inline MC_Base_Particle::MC_Base_Particle( ) :
        coordinate(),
        velocity(),
        kinetic_energy(0.0),
        weight(0.0),
        time_to_census(0.0),
        age(0.0),
        num_mean_free_paths(0.0),
        num_segments(0.0),
        random_number_seed((uint64_t)0),
        identifier((uint64_t)0),
        last_event(MC_Tally_Event::Census),
        num_collisions(0),
        breed(0),
        domain(0),
        cell(0)
{
}


//----------------------------------------------------------------------------------------------------------------------
// Constructor from a base particle type.
//----------------------------------------------------------------------------------------------------------------------
inline MC_Base_Particle::MC_Base_Particle(const MC_Base_Particle &particle)
{
    coordinate          = particle.coordinate;
    velocity            = particle.velocity;
    kinetic_energy      = particle.kinetic_energy;
    weight              = particle.weight;
    time_to_census      = particle.time_to_census;
    age                 = particle.age;
    num_mean_free_paths = particle.num_mean_free_paths;
    num_segments        = particle.num_segments;
    random_number_seed  = particle.random_number_seed;
    identifier          = particle.identifier;
    last_event          = particle.last_event;
    num_collisions      = particle.num_collisions;
    breed               = particle.breed;
    domain              = particle.domain;
    cell                = particle.cell;
}

//----------------------------------------------------------------------------------------------------------------------
// Constructor from a particle type.
//----------------------------------------------------------------------------------------------------------------------
inline MC_Base_Particle::MC_Base_Particle(const MC_Particle &particle)
{
    coordinate          = particle.coordinate;
    velocity            = particle.velocity;
    kinetic_energy      = particle.kinetic_energy;
    weight              = particle.weight;
    time_to_census      = particle.time_to_census;
    age                 = particle.age;
    num_mean_free_paths = particle.num_mean_free_paths;
    num_segments        = particle.num_segments;
    random_number_seed  = particle.random_number_seed;
    identifier          = particle.identifier;
    last_event          = particle.last_event;
    num_collisions      = particle.num_collisions;
    breed               = particle.breed;
    domain              = particle.domain;
    cell                = particle.cell;
}


//----------------------------------------------------------------------------------------------------------------------
// The assignment operator.
// Copies a given (rhs) particle replacing this (lhs) particle..
//----------------------------------------------------------------------------------------------------------------------
inline MC_Base_Particle& MC_Base_Particle::operator= (const MC_Particle &particle)
{
    coordinate = particle.coordinate;
    velocity = particle.velocity;
    kinetic_energy = particle.kinetic_energy;
    weight = particle.weight;
    time_to_census = particle.time_to_census;
    age = particle.age;
    num_mean_free_paths = particle.num_mean_free_paths;
    num_segments = particle.num_segments;
    random_number_seed = particle.random_number_seed;
    identifier = particle.identifier;
    last_event = particle.last_event;
    num_collisions = particle.num_collisions;
    breed = particle.breed;
    domain = particle.domain;
    cell = particle.cell;

    return *this;
}


//----------------------------------------------------------------------------------------------------------------------
// MC_Particle Constructor.
//----------------------------------------------------------------------------------------------------------------------
__host__ __device__ inline MC_Particle::MC_Particle()
   : coordinate(),
     velocity(),
     direction_cosine(),
     kinetic_energy(0.0),
     weight(0.0),
     time_to_census(0.0),
     totalCrossSection(0.0),
     age(0.0),
     num_mean_free_paths(0.0),
     mean_free_path(0.0),
     segment_path_length(0.0),
     random_number_seed((uint64_t)0),
     identifier( (uint64_t)0),
     last_event(MC_Tally_Event::Census),
     num_collisions (0),
     num_segments(0.0),

     breed(0),
     energy_group(0),
     domain(0),
     cell(0),
     facet(0),
     normal_dot(0.0)
{
}


//----------------------------------------------------------------------------------------------------------------------
// MC_Particle Constructor.
//----------------------------------------------------------------------------------------------------------------------
inline MC_Particle::MC_Particle( const MC_Base_Particle &from_particle )
   : coordinate(from_particle.coordinate),
     velocity(from_particle.velocity),
     direction_cosine(), // define this from velocity in body of this function
     kinetic_energy(from_particle.kinetic_energy),

     weight(from_particle.weight),
     time_to_census(from_particle.time_to_census),
     totalCrossSection(0),
     age(from_particle.age),
     num_mean_free_paths(from_particle.num_mean_free_paths),

     mean_free_path(0.0),
     segment_path_length(0.0),

     random_number_seed(from_particle.random_number_seed),
     identifier( from_particle.identifier ),
     last_event(from_particle.last_event),

     num_collisions (from_particle.num_collisions),
     num_segments(from_particle.num_segments),


     breed(from_particle.breed),
     energy_group(0),
     domain(from_particle.domain),
     cell(from_particle.cell),
     facet(0),
     normal_dot(0.0)
{
    double speed = from_particle.velocity.Length();

    if ( speed > 0 )
    {
        double factor = 1.0/speed;
        direction_cosine.alpha = factor * from_particle.velocity.x;
        direction_cosine.beta  = factor * from_particle.velocity.y;
        direction_cosine.gamma = factor * from_particle.velocity.z;
    }
    else
    {
        qs_assert(false);
    }
}


//----------------------------------------------------------------------------------------------------------------------
//  Copy_From_Base copies a particle from a base into this version
//----------------------------------------------------------------------------------------------------------------------
inline void MC_Particle::Copy_From_Base( const MC_Base_Particle &from_particle)
{
    this->coordinate          = from_particle.coordinate;
    this->velocity.x          = from_particle.velocity.x;
    this->velocity.y          = from_particle.velocity.y;
    this->velocity.z          = from_particle.velocity.z;
    this->kinetic_energy      = from_particle.kinetic_energy;
    this->weight              = from_particle.weight;
    this->time_to_census      = from_particle.time_to_census;
    this->age                 = from_particle.age;
    this->num_mean_free_paths = from_particle.num_mean_free_paths;
    this->random_number_seed  = from_particle.random_number_seed;
    this->identifier          = from_particle.identifier;
    this->last_event          = from_particle.last_event;

    this->num_collisions      = from_particle.num_collisions;
    this->num_segments        = from_particle.num_segments;

    this->breed               = from_particle.breed;
    this->domain              = from_particle.domain;
    this->cell                = from_particle.cell;
}

//----------------------------------------------------------------------------------------------------------------------
//  Print the input particle to a string.
//----------------------------------------------------------------------------------------------------------------------
inline void MC_Base_Particle::Copy_Particle_Base_To_String(std::string &output_string) const
{
    MC_Particle to_particle( *this );

    to_particle.Copy_Particle_To_String(output_string);
}


#endif

