//! Hello Swarm Example (`W10-03`, `W11-04`).
//!
//! Demonstrates basic `SwarmCluster` configuration, local `LinearModel` + `MseLoss` + `SgdOptimizer`
//! training, and `SwarmTorchCheckpointV1` SafeTensors v1 binary serialization.

use swarm_torch::core::traits::{SwarmModel, SwarmOptimizer};
use swarm_torch::models::loss::MseLoss;
use swarm_torch::models::optim::SgdOptimizer;
use swarm_torch::models::safetensors_v1::{
    export_safetensors_checkpoint_v1, import_safetensors_checkpoint_v1, ModelTensorSchemaV1,
    SafeTensorsLimitsV1,
};
use swarm_torch::models::simple::LinearModel;
use swarm_torch::prelude::*;
use swarm_torch::SwarmCluster;

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("SwarmTorch Hello Swarm Example (v0.1.0)");
    println!("=======================================");

    // 1. Validate and build a SwarmCluster configuration
    let config = SwarmCluster::builder()
        .topology(Topology::gossip(2))
        .consensus(RobustAggregation::CenteredClipping {
            tau: 1.0,
            max_iterations: 3,
        })
        .max_rounds(50)
        .convergence_threshold(0.005)
        .try_build()?;

    let peer_id = PeerId::new([1u8; 32]);
    let cluster = SwarmCluster::new(config, peer_id);
    println!(
        "Cluster initialized: peer={:?}, max_rounds={}",
        cluster.local_peer(),
        cluster.config().max_rounds
    );

    // 2. Run a local analytical training step on a 4 -> 2 LinearModel
    let mut model = LinearModel::new(4, 2)?.with_random_init(42);
    let mut optimizer: SgdOptimizer<LinearModel> = SgdOptimizer::with_hyperparams(0.05, 0.9, 1e-4)?;
    let input = [1.0f32, -0.5, 0.25, 0.75];
    let target = [0.5f32, -0.25];

    let preds = model.forward(&input)?;
    let mut d_preds = [0.0f32; 2];
    let initial_loss = MseLoss::compute_and_grad(preds.as_slice(), &target, &mut d_preds)?;

    let mut param_grads = [0.0f32; 10];
    model.backward_into(&input, &d_preds, &mut param_grads)?;
    optimizer.step(&mut model, &param_grads)?;

    let preds_after = model.forward(&input)?;
    let post_step_loss = MseLoss::compute_and_grad(preds_after.as_slice(), &target, &mut d_preds)?;
    println!(
        "Local SGD step complete: loss {:.6} -> {:.6}",
        initial_loss, post_step_loss
    );

    // 3. Export and verify a SwarmTorchCheckpointV1 SafeTensors v1 binary payload
    let schema = ModelTensorSchemaV1::for_linear(4, 2)?;
    let limits = SafeTensorsLimitsV1::default();
    let checkpoint_bytes = export_safetensors_checkpoint_v1(&model, &schema, &limits)?;

    let mut restored = LinearModel::new(4, 2)?;
    let schema_hash =
        import_safetensors_checkpoint_v1(&mut restored, &schema, &checkpoint_bytes, &limits)?;
    assert_eq!(restored.parameters(), model.parameters());
    println!(
        "SafeTensors v1 checkpoint verified ({} bytes, schema_hash={:02x}{:02x}..)",
        checkpoint_bytes.len(),
        schema_hash[0],
        schema_hash[1]
    );

    Ok(())
}
