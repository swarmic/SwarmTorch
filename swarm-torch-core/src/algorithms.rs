//! Swarm optimization algorithms
//!
//! This module provides implementations of swarm intelligence algorithms
//! for distributed optimization.

/// Particle Swarm Optimization (PSO) configuration
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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

impl Topology {
    /// Create a gossip topology with specified fanout
    pub fn gossip(fanout: usize) -> Self {
        Self::Gossip { fanout }
    }

    /// Create a hierarchical topology with specified layers
    pub fn hierarchical(layers: usize) -> Self {
        Self::Hierarchical { layers }
    }
}
