use super::*;

#[test]
fn linear_model_new_rejects_oversized_dims() {
    assert!(matches!(
        LinearModel::new(100, 100),
        Err(ModelError::InvalidDimensions)
    ));
    assert!(matches!(
        LinearModel::new(8, 20),
        Err(ModelError::InvalidDimensions)
    ));
    assert!(LinearModel::new(8, 16).is_ok());
}

#[test]
fn linear_model_parameters_include_bias() {
    let model = LinearModel::new(8, 16).unwrap();
    assert_eq!(model.parameters().len(), 8 * 16 + 16);
}

#[test]
fn linear_model_load_parameters_roundtrip_preserves_bias_and_rejects_non_finite() {
    let mut model = LinearModel::new(4, 3).unwrap();
    let mut source = [0.0f32; 15];
    for (i, slot) in source.iter_mut().enumerate() {
        *slot = i as f32 + 0.5;
    }
    model.load_parameters(&source).unwrap();
    assert_eq!(model.parameters(), &source[..]);

    let mut bad = source;
    bad[2] = f32::NAN;
    assert_eq!(model.load_parameters(&bad), Err(Error::NanDetected));
    assert_eq!(
        model.parameters(),
        &source[..],
        "failed load must not mutate model"
    );
}

#[test]
fn linear_model_forward_computes_affine_projection_and_validates_inputs() {
    let mut model = LinearModel::new(2, 2).unwrap();
    // W = [[1.0, 2.0], [3.0, 4.0]], b = [0.5, -0.5]
    model
        .load_parameters(&[1.0, 2.0, 3.0, 4.0, 0.5, -0.5])
        .unwrap();
    let out = model.forward(&[2.0, 3.0]).unwrap();
    assert_eq!(out.len(), 2);
    assert!((out[0] - 8.5).abs() < 1e-6);
    assert!((out[1] - 17.5).abs() < 1e-6);

    assert_eq!(
        model.forward(&[1.0]),
        Err(ModelError::InputDimensionMismatch {
            expected: 2,
            got: 1
        })
    );
    assert_eq!(
        model.forward(&[1.0, f32::NAN]),
        Err(ModelError::NonFiniteInput)
    );
}

#[test]
fn mlp_model_contiguous_parameters_cover_both_layers() {
    let mut mlp = MlpModel::new(3, 4, 2).unwrap().with_random_init(77);
    let expected_l1 = 3 * 4 + 4;
    let expected_l2 = 4 * 2 + 2;
    assert_eq!(mlp.layer1_param_count(), expected_l1);
    assert_eq!(mlp.layer2_param_count(), expected_l2);
    assert_eq!(mlp.parameters().len(), expected_l1 + expected_l2);

    // Mutating the tail of parameters_mut() must directly update layer 2 bias.
    let last_idx = mlp.parameters().len() - 1;
    mlp.parameters_mut()[last_idx] = 0.75;
    assert_eq!(mlp.layer2_parameters().last().copied(), Some(0.75));

    let out = mlp.forward(&[0.5, -0.25, 1.0]).unwrap();
    assert_eq!(out.len(), 2);
}

#[test]
fn linear_model_init_is_seed_deterministic() {
    let a = LinearModel::new(8, 16).unwrap().with_random_init(42);
    let b = LinearModel::new(8, 16).unwrap().with_random_init(42);
    let c = LinearModel::new(8, 16).unwrap().with_random_init(43);

    assert_eq!(a.parameters(), b.parameters());
    assert_ne!(a.parameters(), c.parameters());
}

#[test]
fn linear_model_init_distribution_sanity() {
    let model = LinearModel::new(8, 16).unwrap().with_random_init(1337);
    let params = model.parameters();
    let n = params.len() as f32;
    let mean = params.iter().copied().sum::<f32>() / n;
    let var = params
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / n;

    assert!(mean.abs() < 0.1, "mean drift too large: {mean}");
    assert!(var > 1.0e-4, "variance too small: {var}");
    assert!(var < 0.2, "variance too large: {var}");
}

#[test]
fn linear_model_const_generic_capacity_bounds() {
    type TinyLinear = LinearModelWithCapacity<32, 4>;

    assert!(TinyLinear::new(7, 4).is_ok(), "7*4=28 fits");
    assert!(
        matches!(TinyLinear::new(8, 4), Err(ModelError::InvalidDimensions)),
        "8*4=32 exceeds MAX_PARAMS-MAX_OUTPUT (28)"
    );
    assert!(
        matches!(TinyLinear::new(7, 5), Err(ModelError::InvalidDimensions)),
        "output_dim exceeds MAX_OUTPUT"
    );
}

#[test]
fn linear_model_rejects_output_dim_exceeding_param_capacity() {
    type WeirdLinear = LinearModelWithCapacity<8, 32>;
    assert!(
        matches!(WeirdLinear::new(0, 16), Err(ModelError::InvalidDimensions)),
        "output region must fit inside MAX_PARAMS even when MAX_OUTPUT is larger"
    );
}
