//! Model-neutral decoder-only transformer runtime for NNIS.
//!
//! This crate owns model configuration, device weight graphs and the exact
//! model-level CUDA operations that are not already covered by `nnis-kernels`.
//! It intentionally does not claim Hugging Face compatibility.

mod config;
mod kernels;
mod weights;

pub use config::{Activation, GenerationConfig, ModelConfig, WeightDType};
pub use kernels::F32DecoderKernels;
pub use weights::{
    DecoderLayerWeights, DeviceTensor, MatrixWeight, ModelWeights, VectorWeight,
};
