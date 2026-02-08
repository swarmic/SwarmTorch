//! Hello Swarm Example
//!
//! Demonstrates basic swarm cluster setup and training.

use swarm_torch::prelude::*;
use swarm_torch::SwarmCluster;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SwarmTorch Hello Swarm Example");
    println!("==============================");

    // Create a basic configuration
    let config = SwarmCluster::builder()
        .topology(Topology::gossip(2))
        .consensus(RobustAggregation::TrimmedMean { trim_ratio: 0.2 })
        .max_rounds(100)
        .convergence_threshold(0.01)
        .build();

    println!("Configuration:");
    println!("  Max rounds: {}", config.max_rounds);
    println!("  Convergence threshold: {}", config.convergence_threshold);

    // Create a local peer ID
    let peer_id = PeerId::new([1u8; 32]);

    // Create the cluster
    let cluster = SwarmCluster::new(config, peer_id);

    println!("\nCluster created with peer ID: {:?}", cluster.local_peer());
    println!("\nSwarmTorch is ready for distributed learning!");

    Ok(())
}
