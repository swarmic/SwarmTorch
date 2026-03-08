//! Swarm optimization algorithms
//!
//! This module provides implementations of swarm intelligence algorithms
//! for distributed optimization.

/// Particle Swarm Optimization (PSO) configuration
#[derive(Debug, Clone)]
#[deprecated(
    since = "0.1.0-alpha.6x",
    note = "PSO execution is not implemented. This type will be feature-gated behind \
            `experimental-pso` in a future breaking-change release."
)]
pub struct ParticleSwarmConfig {
    /// Number of particles in the swarm
    pub num_particles: usize,
    /// Inertia weight (momentum)
    pub inertia: f32,
    /// Cognitive coefficient (attraction to personal best)
    pub cognitive: f32,
    /// Social coefficient (attraction to global best)
    pub social: f32,
    /// Maximum velocity
    pub max_velocity: f32,
}

#[allow(deprecated)]
impl Default for ParticleSwarmConfig {
    fn default() -> Self {
        Self {
            num_particles: 50,
            inertia: 0.7,
            cognitive: 1.5,
            social: 1.5,
            max_velocity: 1.0,
        }
    }
}

/// Particle state in PSO
#[derive(Debug, Clone)]
#[deprecated(
    since = "0.1.0-alpha.6x",
    note = "PSO execution is not implemented. See `ParticleSwarmConfig` deprecation note."
)]
pub struct Particle {
    /// Current position (parameters)
    pub position: [f32; 128], // Fixed size for no_std
    /// Current velocity
    pub velocity: [f32; 128],
    /// Personal best position
    pub best_position: [f32; 128],
    /// Personal best fitness
    pub best_fitness: f32,
}

#[allow(deprecated)]
impl Default for Particle {
    fn default() -> Self {
        Self {
            position: [0.0; 128],
            velocity: [0.0; 128],
            best_position: [0.0; 128],
            best_fitness: f32::NEG_INFINITY,
        }
    }
}

/// Ant Colony Optimization (ACO) configuration
#[derive(Debug, Clone)]
pub struct AntColonyConfig {
    /// Number of ants
    pub num_ants: usize,
    /// Pheromone evaporation rate
    pub evaporation_rate: f32,
    /// Pheromone deposit factor
    pub deposit_factor: f32,
    /// Exploration vs exploitation balance
    pub alpha: f32,
    /// Heuristic information weight
    pub beta: f32,
}

impl Default for AntColonyConfig {
    fn default() -> Self {
        Self {
            num_ants: 50,
            evaporation_rate: 0.1,
            deposit_factor: 1.0,
            alpha: 1.0,
            beta: 2.0,
        }
    }
}

/// Firefly Algorithm configuration
#[derive(Debug, Clone)]
pub struct FireflyConfig {
    /// Number of fireflies
    pub num_fireflies: usize,
    /// Light absorption coefficient
    pub gamma: f32,
    /// Attractiveness at distance 0
    pub beta_0: f32,
    /// Randomization parameter
    pub alpha: f32,
}

impl Default for FireflyConfig {
    fn default() -> Self {
        Self {
            num_fireflies: 50,
            gamma: 1.0,
            beta_0: 1.0,
            alpha: 0.2,
        }
    }
}

/// Swarm topology configuration
#[derive(Debug, Clone, PartialEq)]
pub enum Topology {
    /// Full mesh - every node connected to every other
    FullMesh,
    /// Ring topology - each node connected to neighbors
    Ring,
    /// Random gossip with specified fanout
    Gossip { fanout: usize },
    /// Hierarchical with specified number of layers
    Hierarchical { layers: usize },
    /// Star topology with central coordinator
    Star,
}

impl Default for Topology {
    fn default() -> Self {
        Self::Gossip { fanout: 3 }
    }
}

/// Maximum gossip fanout (M-14).
pub const MAX_GOSSIP_FANOUT: usize = 64;
/// Maximum number of hierarchy layers (M-14).
pub const MAX_HIERARCHY_LAYERS: usize = 16;

/// Topology configuration validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    /// Gossip fanout must be non-zero.
    FanoutZero,
    /// Gossip fanout exceeds maximum.
    FanoutTooLarge { fanout: usize, max: usize },
    /// Hierarchy layers must be non-zero.
    LayersZero,
    /// Hierarchy layers exceed maximum.
    LayersTooLarge { layers: usize, max: usize },
}

impl core::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::FanoutZero => write!(f, "gossip fanout must be non-zero"),
            Self::FanoutTooLarge { fanout, max } => {
                write!(f, "gossip fanout {fanout} exceeds maximum {max}")
            }
            Self::LayersZero => write!(f, "hierarchy layers must be non-zero"),
            Self::LayersTooLarge { layers, max } => {
                write!(f, "hierarchy layers {layers} exceeds maximum {max}")
            }
        }
    }
}

impl Topology {
    /// Create a gossip topology with specified fanout
    pub fn gossip(fanout: usize) -> Self {
        Self::Gossip { fanout }
    }

    /// Create a gossip topology with validated fanout (M-14).
    ///
    /// Returns `Err` if `fanout` is 0 or exceeds [`MAX_GOSSIP_FANOUT`].
    pub fn try_gossip(fanout: usize) -> core::result::Result<Self, TopologyError> {
        if fanout == 0 {
            return Err(TopologyError::FanoutZero);
        }
        if fanout > MAX_GOSSIP_FANOUT {
            return Err(TopologyError::FanoutTooLarge {
                fanout,
                max: MAX_GOSSIP_FANOUT,
            });
        }
        Ok(Self::Gossip { fanout })
    }

    /// Create a hierarchical topology with specified layers
    pub fn hierarchical(layers: usize) -> Self {
        Self::Hierarchical { layers }
    }

    /// Create a hierarchical topology with validated layers (M-14).
    ///
    /// Returns `Err` if `layers` is 0 or exceeds [`MAX_HIERARCHY_LAYERS`].
    pub fn try_hierarchical(layers: usize) -> core::result::Result<Self, TopologyError> {
        if layers == 0 {
            return Err(TopologyError::LayersZero);
        }
        if layers > MAX_HIERARCHY_LAYERS {
            return Err(TopologyError::LayersTooLarge {
                layers,
                max: MAX_HIERARCHY_LAYERS,
            });
        }
        Ok(Self::Hierarchical { layers })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_gossip_fanout_zero_rejected() {
        assert_eq!(Topology::try_gossip(0), Err(TopologyError::FanoutZero));
    }

    #[test]
    fn try_gossip_fanout_too_large_rejected() {
        assert_eq!(
            Topology::try_gossip(65),
            Err(TopologyError::FanoutTooLarge {
                fanout: 65,
                max: MAX_GOSSIP_FANOUT,
            })
        );
    }

    #[test]
    fn try_gossip_valid_passes() {
        assert!(Topology::try_gossip(4).is_ok());
        assert!(Topology::try_gossip(64).is_ok()); // max boundary
    }

    #[test]
    fn try_hierarchical_layers_zero_rejected() {
        assert_eq!(
            Topology::try_hierarchical(0),
            Err(TopologyError::LayersZero)
        );
    }

    #[test]
    fn try_hierarchical_layers_too_large_rejected() {
        assert_eq!(
            Topology::try_hierarchical(17),
            Err(TopologyError::LayersTooLarge {
                layers: 17,
                max: MAX_HIERARCHY_LAYERS,
            })
        );
    }

    #[test]
    fn try_hierarchical_valid_passes() {
        assert!(Topology::try_hierarchical(3).is_ok());
        assert!(Topology::try_hierarchical(16).is_ok()); // max boundary
    }
}
