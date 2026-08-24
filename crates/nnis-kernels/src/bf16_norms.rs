//! Packed-bf16 row normalization through the validated f32 families.
//!
//! Pure composition under the crate-wide bf16 policy: exact widening of
//! the packed input, one dispatched f32 normalize (fused when the row fits
//! dynamic shared memory, staged pipeline otherwise), and one
//! round-to-nearest-even narrowing of the output. No new CUDA kernels -
//! the arithmetic is bit-identical to running the f32 family on widened
//! buffers, which tests assert after a host RNE narrowing, alongside f64
//! oracle tolerances that carry the output quantization term.

use crate::{Bf16Elementwise, F32LayerNorm, F32RmsNorm};
use nnis_rt::{DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

/// Validate shared shape/context preconditions for both wrappers.
fn validate(
    conversions: &Bf16Elementwise,
    stream: &Stream,
    input: &DeviceBuffer<u16>,
    output: &DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
    family: &str,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(NnisError::invalid_input(format!(
            "bf16 {family} requires non-empty rows and columns; \
             got rows={rows}, cols={cols}"
        )));
    }
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| NnisError::invalid_input("bf16 norm shape overflows usize"))?;
    if input.len() != expected || output.len() != expected {
        return Err(NnisError::invalid_input(format!(
            "bf16 {family} buffers have {}/{} elements; shape ({rows}, {cols}) requires {expected}",
            input.len(),
            output.len()
        )));
    }
    // SAFETY-free context plumbing: both families expose their kernels'
    // context; equality with the stream keeps every launch on one device.
    let _ = conversions;
    let context = stream.ctx();
    if !Arc::ptr_eq(context, input.ctx()) || !Arc::ptr_eq(context, output.ctx()) {
        return Err(NnisError::invalid_input(format!(
            "bf16 {family} stream and buffers must share one context"
        )));
    }
    Ok(())
}

/// Normalize packed-bf16 rows with RMS statistics (`x / rms(x) * gamma`)
/// and wait once.
#[allow(clippy::too_many_arguments)]
pub fn bf16_rms_normalize_rows_dispatched(
    conversions: &Bf16Elementwise,
    rms_norm: &F32RmsNorm,
    stream: &Stream,
    input: &DeviceBuffer<u16>,
    output: &DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
    epsilon: f32,
    gamma: f32,
) -> Result<()> {
    validate(conversions, stream, input, output, rows, cols, "rms norm")?;
    let context = stream.ctx();
    let wide_input = DeviceBuffer::<f32>::new(context, rows * cols)?;
    let wide_output = DeviceBuffer::<f32>::new(context, rows * cols)?;
    conversions.widen(stream, input, &wide_input)?;
    rms_norm.normalize_rows_dispatched(
        stream,
        &wide_input,
        &wide_output,
        rows,
        cols,
        epsilon,
        gamma,
    )?;
    conversions.narrow(stream, &wide_output, output)
}

/// Normalize packed-bf16 rows with full layer statistics
/// (`(x - mean) / sqrt(var + eps) * gamma + beta`) and wait once.
#[allow(clippy::too_many_arguments)]
pub fn bf16_layer_normalize_rows_dispatched(
    conversions: &Bf16Elementwise,
    layer_norm: &F32LayerNorm,
    stream: &Stream,
    input: &DeviceBuffer<u16>,
    output: &DeviceBuffer<u16>,
    rows: usize,
    cols: usize,
    epsilon: f32,
    gamma: f32,
    beta: f32,
) -> Result<()> {
    validate(conversions, stream, input, output, rows, cols, "layer norm")?;
    let context = stream.ctx();
    let wide_input = DeviceBuffer::<f32>::new(context, rows * cols)?;
    let wide_output = DeviceBuffer::<f32>::new(context, rows * cols)?;
    conversions.widen(stream, input, &wide_input)?;
    layer_norm.normalize_rows_dispatched(
        stream,
        &wide_input,
        &wide_output,
        rows,
        cols,
        epsilon,
        gamma,
        beta,
    )?;
    conversions.narrow(stream, &wide_output, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::{bf16_bits_to_f32, f32_to_bf16_rne, gpu_context, Context};
    use std::sync::Arc;

    const SHAPES: &[(usize, usize)] = &[(1, 1), (3, 17), (17, 64), (2, 16_384)];

    fn host_values(len: usize) -> Vec<f32> {
        (0..len)
            .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
            .collect()
    }

    fn to_bits(values: &[f32]) -> Vec<u16> {
        values.iter().copied().map(f32_to_bf16_rne).collect()
    }

    fn widened(bits: &[u16]) -> Vec<f32> {
        bits.iter().copied().map(bf16_bits_to_f32).collect()
    }

    /// Tolerance carries the output bf16 quantization term (half-ulp below
    /// 2^-8 relative) over the f32 statistic chain error.
    fn oracle_tolerance(expected: f64) -> f64 {
        5.0e-3_f64.max(expected.abs() * 8.0e-3)
    }

    /// Asserts bf16 output bits equal widen->f32-family->narrow replayed
    /// on device, and that values track the f64 oracle inside tolerances.
    #[allow(clippy::too_many_arguments)]
    fn assert_bit_exact_and_close<const WITH_BETA: bool>(
        actual_bits: &[u16],
        f32_reference: &[f32],
        wide_input: &[f32],
        _rows: usize,
        cols: usize,
        epsilon: f64,
        gamma: f64,
        beta: f64,
        context: &str,
    ) {
        assert_eq!(actual_bits.len(), f32_reference.len());
        for index in 0..actual_bits.len() {
            assert_eq!(
                actual_bits[index],
                f32_to_bf16_rne(f32_reference[index]),
                "{context} bit mismatch at {index}"
            );
            let row = index / cols;
            let col = index % cols;
            let mut mean = 0.0_f64;
            if WITH_BETA {
                let total: f64 = (0..cols)
                    .map(|c| f64::from(wide_input[row * cols + c]))
                    .sum();
                mean = total / cols as f64;
            }
            let mut variance = 0.0_f64;
            for c in 0..cols {
                let value = f64::from(wide_input[row * cols + c]) - mean;
                variance += value * value;
            }
            let denominator = (variance / cols as f64 + epsilon).sqrt();
            let normalized = (f64::from(wide_input[index]) - mean) / denominator * gamma + beta;
            let actual = f64::from(bf16_bits_to_f32(actual_bits[index]));
            let tolerance = oracle_tolerance(normalized);
            assert!(
                (actual - normalized).abs() <= tolerance,
                "{context} mismatch at ({row}, {col}): {actual} vs {normalized}, \
                 tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn bf16_norms_bit_match_f32_family_and_f64_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = nnis_jit::JitCompiler::new();
        let conversions = Bf16Elementwise::load(&context, &compiler).unwrap();
        let rms_norm = F32RmsNorm::load(&context, &compiler).unwrap();
        let layer_norm = F32LayerNorm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        let epsilon = 1.0e-6_f32;
        let gamma = 1.25_f32;
        let beta = -0.5_f32;

        for &(rows, cols) in SHAPES {
            // cols=16384 exceeds the fused shared-memory budget at the
            // default block width, exercising the staged fallback too.
            let bits = to_bits(&host_values(rows * cols));
            let input = DeviceBuffer::from_host(&context, &stream, &bits).unwrap();
            let wide_host = widened(&bits);
            let wide_input = DeviceBuffer::from_host(&context, &stream, &wide_host).unwrap();

            let poisoned = vec![0xFFFF_u16; rows * cols];
            let rms_output = DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
            let layer_output =
                DeviceBuffer::from_host(&context, &stream, &poisoned.clone()).unwrap();
            let f32_rms_output =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; rows * cols]).unwrap();
            let f32_layer_output =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; rows * cols]).unwrap();

            bf16_rms_normalize_rows_dispatched(
                &conversions,
                &rms_norm,
                &stream,
                &input,
                &rms_output,
                rows,
                cols,
                epsilon,
                gamma,
            )
            .unwrap();
            rms_norm
                .normalize_rows_dispatched(
                    &stream,
                    &wide_input,
                    &f32_rms_output,
                    rows,
                    cols,
                    epsilon,
                    gamma,
                )
                .unwrap();

            bf16_layer_normalize_rows_dispatched(
                &conversions,
                &layer_norm,
                &stream,
                &input,
                &layer_output,
                rows,
                cols,
                epsilon,
                gamma,
                beta,
            )
            .unwrap();
            layer_norm
                .normalize_rows_dispatched(
                    &stream,
                    &wide_input,
                    &f32_layer_output,
                    rows,
                    cols,
                    epsilon,
                    gamma,
                    beta,
                )
                .unwrap();

            assert_bit_exact_and_close::<false>(
                &rms_output.to_vec(&stream).unwrap(),
                &f32_rms_output.to_vec(&stream).unwrap(),
                &wide_host,
                rows,
                cols,
                f64::from(epsilon),
                f64::from(gamma),
                0.0,
                &format!("bf16 rms ({rows},{cols})"),
            );
            assert_bit_exact_and_close::<true>(
                &layer_output.to_vec(&stream).unwrap(),
                &f32_layer_output.to_vec(&stream).unwrap(),
                &wide_host,
                rows,
                cols,
                f64::from(epsilon),
                f64::from(gamma),
                f64::from(beta),
                &format!("bf16 layer ({rows},{cols})"),
            );
        }
    }

    #[test]
    fn bf16_norms_reject_invalid_shapes_and_contexts_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let other_context: Arc<Context> = {
            let device = nnis_rt::Device::first().unwrap();
            nnis_rt::Context::new(&device).unwrap()
        };
        let compiler = nnis_jit::JitCompiler::new();
        let conversions = Bf16Elementwise::load(&context, &compiler).unwrap();
        let rms_norm = F32RmsNorm::load(&context, &compiler).unwrap();
        let layer_norm = F32LayerNorm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        // Zero columns is outside the contract.
        let empty = DeviceBuffer::<u16>::new(&context, 0).unwrap();
        let error = bf16_rms_normalize_rows_dispatched(
            &conversions,
            &rms_norm,
            &stream,
            &empty,
            &empty,
            1,
            0,
            1.0e-6,
            1.0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("non-empty"), "{error}");

        // Short output rejected with the required size in the message.
        let input = DeviceBuffer::<u16>::new(&context, 17).unwrap(); // 1 x 17
        let short_output = DeviceBuffer::<u16>::new(&context, 16).unwrap();
        let error = bf16_rms_normalize_rows_dispatched(
            &conversions,
            &rms_norm,
            &stream,
            &input,
            &short_output,
            1,
            17,
            1.0e-6,
            1.0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("requires 17"), "{error}");

        // Buffers from a foreign context are rejected before any launch.
        let foreign_input = DeviceBuffer::<u16>::new(&other_context, 17).unwrap();
        let foreign_output = DeviceBuffer::<u16>::new(&other_context, 17).unwrap();
        let error = bf16_layer_normalize_rows_dispatched(
            &conversions,
            &layer_norm,
            &stream,
            &foreign_input,
            &foreign_output,
            1,
            17,
            1.0e-6,
            1.0,
            0.0,
        )
        .unwrap_err();
        assert!(error.to_string().contains("share one context"), "{error}");
    }
}
