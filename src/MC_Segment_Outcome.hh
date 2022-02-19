#ifndef MC_SEGMENT_OUTCOME_INCLUDE
#define MC_SEGMENT_OUTCOME_INCLUDE

#include "MacroscopicCrossSection.hh"
#include "MC_Nearest_Facet.hh"
#include "MCT.hh"
#include "PhysicalConstants.hh"

class MC_Particle;
class MC_Vector;
struct Device;


struct MC_Segment_Outcome_type
{
    public:
    enum Enum
    {
        Initialize                    = -1,
        Collision                     = 0,
        Facet_Crossing                = 1,
        Census                        = 2,
        Max_Number                    = 3
    };
};


struct MC_Collision_Event_Return
{
    public:
    enum Enum
    {
        Stop_Tracking     = 0,
        Continue_Tracking = 1,
        Continue_Collision = 2
    };
};

#include "DeclareMacro.hh"

static inline unsigned int MC_Find_Min(const double *__restrict__ const array)
{
    double min = array[0];
    int    min_index = 0;
    for (int element_index = 1; element_index < 3; ++element_index)
    {
        if ( array[element_index] < min )
        {
            min = array[element_index];
            min_index = element_index;
        }
    }

    return min_index;
}

//--------------------------------------------------------------------------------------------------
//  Routine MC_Segment_Outcome determines whether the next segment of the particle's trajectory will result in:
//    (i) collision within the current cell,
//   (ii) exiting from the current cell, or
//  (iii) census at the end of the time step.
//--------------------------------------------------------------------------------------------------

static inline MC_Segment_Outcome_type::Enum MC_Segment_Outcome(Device &device, MC_Particle &mc_particle)
{
    // initialize distances to large number
    double distance[3];
    distance[0] = distance[1] = distance[2] = 1e80;

    // Calculate the particle speed
    const double particle_speed = mc_particle.velocity.Length();

    // Force collision if a census event narrowly preempts a collision
    int force_collision = 0 ;
    if ( mc_particle.num_mean_free_paths < 0.0 )
    {
        force_collision = 1 ;

        if ( mc_particle.num_mean_free_paths > -900.0 ) abort();

        mc_particle.num_mean_free_paths = PhysicalConstants::_smallDouble;
    }

    // Randomly determine the distance to the next collision
    // based upon the composition of the current cell.
    const double macroscopic_total_cross_section = weightedMacroscopicCrossSection(device, 0,
                             mc_particle.domain, mc_particle.cell, mc_particle.energy_group);

    // Cache the cross section
    mc_particle.totalCrossSection = macroscopic_total_cross_section;
    if (macroscopic_total_cross_section == 0.0)
    {
        mc_particle.mean_free_path = PhysicalConstants::_hugeDouble;
    }
    else
    {
        mc_particle.mean_free_path = 1.0 / macroscopic_total_cross_section;
    }

    if ( mc_particle.num_mean_free_paths == 0.0)
    {
        // Sample the number of mean-free-paths remaining before
        // the next collision from an exponential distribution.
        const double random_number = rngSample(&mc_particle.random_number_seed);

        mc_particle.num_mean_free_paths = -1.0*log(random_number);
    }

    // Calculate the distances to collision, nearest facet, and census.

    // Forced collisions do not need to move far.
    if (force_collision)
    {
        distance[MC_Segment_Outcome_type::Collision] = PhysicalConstants::_smallDouble;
    }
    else
    {
        distance[MC_Segment_Outcome_type::Collision] = mc_particle.num_mean_free_paths*mc_particle.mean_free_path;
    }

    // process census
    distance[MC_Segment_Outcome_type::Census] = particle_speed*mc_particle.time_to_census;


    //  DEBUG  Turn off threshold for now
    static constexpr double distance_threshold = 10.0 * PhysicalConstants::_hugeDouble;
    // Get the current winning distance.
    double current_best_distance = PhysicalConstants::_hugeDouble;

    bool new_segment =  (mc_particle.num_segments == 0 ||
                         mc_particle.last_event == MC_Tally_Event::Collision);

    MC_Location location(mc_particle.Get_Location());

    // Calculate the minimum distance to each facet of the cell.
    MC_Nearest_Facet nearest_facet;
    nearest_facet = MCT_Nearest_Facet(&mc_particle, location, mc_particle.coordinate,
        &mc_particle.direction_cosine, distance_threshold, current_best_distance, new_segment, device);

    mc_particle.normal_dot = nearest_facet.dot_product;

    distance[MC_Segment_Outcome_type::Facet_Crossing] = nearest_facet.distance_to_facet;


    // Get out of here if the tracker failed to bound this particle's volume.
    if (mc_particle.last_event == MC_Tally_Event::Facet_Crossing_Tracking_Error)
    {
        return MC_Segment_Outcome_type::Facet_Crossing;
    }

    // Calculate the minimum distance to the selected events.

    // Force a collision (if required).
    if ( force_collision == 1 )
    {
        distance[MC_Segment_Outcome_type::Facet_Crossing] = PhysicalConstants::_hugeDouble;
        distance[MC_Segment_Outcome_type::Census]         = PhysicalConstants::_hugeDouble;
        distance[MC_Segment_Outcome_type::Collision]      = PhysicalConstants::_tinyDouble ;
    }

    // we choose our segment outcome here
    MC_Segment_Outcome_type::Enum segment_outcome =
        (MC_Segment_Outcome_type::Enum) MC_Find_Min(distance);

    qs_assert(distance[segment_outcome] >= 0);
    mc_particle.segment_path_length = distance[segment_outcome];

    mc_particle.num_mean_free_paths -= mc_particle.segment_path_length / mc_particle.mean_free_path;

    // Before using segment_outcome as an index, verify it is valid
    qs_assert(segment_outcome >= 0);
    qs_assert(segment_outcome < MC_Segment_Outcome_type::Max_Number);

    static constexpr MC_Tally_Event::Enum SegmentOutcome_to_LastEvent[MC_Segment_Outcome_type::Max_Number] =
    {
        MC_Tally_Event::Collision,
        MC_Tally_Event::Facet_Crossing_Transit_Exit,
        MC_Tally_Event::Census,
    };

    mc_particle.last_event = SegmentOutcome_to_LastEvent[segment_outcome];

    // Set the segment path length to be the minimum of
    //   (i)   the distance to collision in the cell, or
    //   (ii)  the minimum distance to a facet of the cell, or
    //   (iii) the distance to census at the end of the time step
    if (segment_outcome == MC_Segment_Outcome_type::Collision)
    {
        mc_particle.num_mean_free_paths = 0.0;
    }
    else if (segment_outcome == MC_Segment_Outcome_type::Facet_Crossing)
    {
        mc_particle.facet = nearest_facet.facet;
    }
    else if (segment_outcome == MC_Segment_Outcome_type::Census)
    {
        mc_particle.time_to_census = MC_MIN(mc_particle.time_to_census, 0.0);
    }

    // If collision was forced, set mc_particle.num_mean_free_paths = 0
    // so that a new value is randomly selected on next pass.
    if (force_collision == 1) { mc_particle.num_mean_free_paths = 0.0; }

    // Do not perform any tallies if the segment path length is zero.
    //   This only introduces roundoff errors.
    if (mc_particle.segment_path_length == 0.0)
    {
        return segment_outcome;
    }

    // Move particle to end of segment, accounting for some physics processes along the segment.

    // Project the particle trajectory along the segment path length.
    mc_particle.Move_Particle(mc_particle.direction_cosine, mc_particle.segment_path_length);

    const double segment_path_time = (mc_particle.segment_path_length/particle_speed);

    // Decrement the time to census and increment age.
    mc_particle.time_to_census -= segment_path_time;
    mc_particle.age += segment_path_time;

    // Ensure mc_particle.time_to_census is non-negative.
    if (mc_particle.time_to_census < 0.0)
    {
        mc_particle.time_to_census = 0.0;
    }

    // Accumulate the particle's contribution to the scalar flux.
    const double value = mc_particle.segment_path_length * mc_particle.weight;
    atomicFetchAdd(device.domains[mc_particle.domain].cells[mc_particle.cell].groupTallies+mc_particle.energy_group,value);

    return segment_outcome;
}

#endif
