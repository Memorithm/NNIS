use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableQueue};
use nnis_cpu::numerical::{
    CpuF32BinaryOp, CpuF32KernelsV1, CpuF32Operation, CPU_F32_CONTRACT_VERSION,
};
use nnis_cpu::{CpuBuffer, CpuDevice};

fn raw(bytes: &[u8], storage: bool) -> CpuBuffer {
    let device = CpuDevice::with_max_buffer_bytes(4096).unwrap();
    let mut usage = BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    if storage {
        usage |= BufferUsages::STORAGE;
    }
    let descriptor = BufferDesc::new(bytes.len() as u64, usage, MemoryClass::Host).unwrap();
    let mut buffer = device.create_buffer(descriptor).unwrap();
    device
        .create_queue()
        .unwrap()
        .write_buffer(&mut buffer, 0, bytes)
        .unwrap();
    buffer
}

fn tensor(values: &[f32]) -> CpuBuffer {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    raw(&bytes, true)
}

fn bytes(buffer: &CpuBuffer) -> Vec<u8> {
    CpuDevice::new()
        .unwrap()
        .create_queue()
        .unwrap()
        .read_buffer(buffer, 0, buffer.len() as u64)
        .unwrap()
}

fn assert_values(buffer: &CpuBuffer, expected: &[f32]) {
    let expected: Vec<u8> = expected
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    assert_eq!(bytes(buffer), expected);
}

fn kernels() -> CpuF32KernelsV1 {
    CpuF32KernelsV1::new(4096).unwrap()
}

#[test]
fn binary_operations_preserve_inputs_and_output_capacity() {
    let left = tensor(&[1.0, -2.0, 0.5]);
    let right = tensor(&[3.0, 4.0, 2.0]);
    let before = bytes(&left);
    let mut output = tensor(&[99.0; 3]);
    let capacity = output.capacity_bytes();
    let report = kernels()
        .binary(CpuF32BinaryOp::Add, &left, &right, &mut output)
        .unwrap();
    assert_values(&output, &[4.0, 2.0, 2.5]);
    assert_eq!(report.schema_version, CPU_F32_CONTRACT_VERSION);
    assert_eq!(report.operation, CpuF32Operation::Add);
    assert_eq!(report.output_values, 3);
    assert_eq!(report.scratch_payload_bytes, 12);
    assert!(report.scratch_capacity_bytes >= report.scratch_payload_bytes);
    kernels()
        .binary(CpuF32BinaryOp::Multiply, &left, &right, &mut output)
        .unwrap();
    assert_values(&output, &[3.0, -8.0, 1.0]);
    assert_eq!(bytes(&left), before);
    assert_eq!(output.capacity_bytes(), capacity);
}

#[test]
fn relu_canonicalizes_both_zeros_and_preserves_positive_subnormal() {
    let input = tensor(&[-3.0, -0.0, 0.0, f32::from_bits(1), 2.0]);
    let mut output = tensor(&[9.0; 5]);
    kernels().relu(&input, &mut output).unwrap();
    assert_values(&output, &[0.0, 0.0, 0.0, f32::from_bits(1), 2.0]);
}

#[test]
fn sum_is_serial_and_does_not_reassociate() {
    let mut output = tensor(&[9.0]);
    kernels()
        .sum(&tensor(&[16_777_216.0, 1.0, -16_777_216.0]), &mut output)
        .unwrap();
    assert_values(&output, &[0.0]);
    kernels()
        .sum(&tensor(&[16_777_216.0, -16_777_216.0, 1.0]), &mut output)
        .unwrap();
    assert_values(&output, &[1.0]);
}

#[test]
fn projection_respects_kn_orientation_on_rectangular_matrix() {
    let input = tensor(&[2.0, -1.0, 3.0]);
    let weights = tensor(&[1.0, 2.0, -1.0, 4.0, 0.5, -2.0]);
    let mut output = tensor(&[99.0; 2]);
    kernels()
        .project_kn(&input, &weights, &mut output, 3, 2)
        .unwrap();
    assert_values(&output, &[4.5, -6.0]);
}

#[test]
fn projection_uses_one_fused_rounding_not_split_multiply_add() {
    let a = f32::from_bits(0x3f80_0001);
    let b = f32::from_bits(0x3f7f_fffe);
    let input = tensor(&[-1.0, a]);
    let weight = tensor(&[1.0, b]);
    let mut output = tensor(&[99.0]);
    kernels()
        .project_kn(&input, &weight, &mut output, 2, 1)
        .unwrap();
    // (1+2^-23)(1-2^-23)-1 = -2^-46, exactly representable.
    assert_values(&output, &[f32::from_bits(0xa880_0000)]);
}

#[test]
fn late_projection_overflow_preserves_entire_output() {
    let input = tensor(&[1.0, f32::MAX]);
    let weight = tensor(&[1.0, 1.0, 0.0, 2.0]);
    let mut output = tensor(&[42.0, -0.0]);
    let before = bytes(&output);
    assert!(kernels()
        .project_kn(&input, &weight, &mut output, 2, 2)
        .is_err());
    assert_eq!(bytes(&output), before);
}

#[test]
fn overflowing_partial_sum_is_not_rescued_by_later_cancellation() {
    let mut output = tensor(&[42.0]);
    assert!(kernels()
        .sum(&tensor(&[f32::MAX, f32::MAX, -f32::MAX]), &mut output)
        .is_err());
    assert_values(&output, &[42.0]);
}

#[test]
fn binary_overflow_is_atomic_even_after_valid_first_element() {
    let mut output = tensor(&[42.0, -0.0]);
    let before = bytes(&output);
    for operation in [CpuF32BinaryOp::Add, CpuF32BinaryOp::Multiply] {
        assert!(kernels()
            .binary(
                operation,
                &tensor(&[1.0, f32::MAX]),
                &tensor(&[2.0, f32::MAX]),
                &mut output
            )
            .is_err());
        assert_eq!(bytes(&output), before);
    }
}

#[test]
fn nonfinite_inputs_are_rejected_including_unselected_gather_values() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let input = tensor(&[1.0, value]);
        let mut vector = tensor(&[42.0, 42.0]);
        let mut scalar = tensor(&[42.0]);
        assert!(kernels().relu(&input, &mut vector).is_err());
        assert!(kernels()
            .binary(CpuF32BinaryOp::Add, &input, &tensor(&[1.0; 2]), &mut vector)
            .is_err());
        assert!(kernels().sum(&input, &mut scalar).is_err());
        assert!(kernels().gather(&input, &[0], &mut scalar).is_err());
        assert!(kernels()
            .project_kn(&tensor(&[1.0]), &input, &mut vector, 1, 2)
            .is_err());
        assert_values(&vector, &[42.0, 42.0]);
        assert_values(&scalar, &[42.0]);
    }
}

#[test]
fn storage_usage_and_whole_word_lengths_are_required() {
    let good = tensor(&[1.0]);
    let missing_storage = raw(&1.0_f32.to_le_bytes(), false);
    let truncated = raw(&[0, 0, 0], true);
    let mut output = tensor(&[42.0]);
    for input in [&missing_storage, &truncated] {
        assert!(kernels().relu(input, &mut output).is_err());
        assert_values(&output, &[42.0]);
    }
    let mut forbidden_output = raw(&42.0_f32.to_le_bytes(), false);
    assert!(kernels().relu(&good, &mut forbidden_output).is_err());
    assert_values(&forbidden_output, &[42.0]);
}

#[test]
fn shapes_and_dimension_overflow_fail_before_output_changes() {
    let input = tensor(&[1.0, 2.0]);
    let weight = tensor(&[1.0, 2.0]);
    let mut output = tensor(&[42.0]);
    for (k, n) in [(0, 1), (1, 0), (usize::MAX, 2), (1, 2), (2, 2)] {
        assert!(kernels()
            .project_kn(&input, &weight, &mut output, k, n)
            .is_err());
        assert_values(&output, &[42.0]);
    }
    assert!(kernels()
        .binary(CpuF32BinaryOp::Add, &input, &tensor(&[1.0]), &mut output)
        .is_err());
    assert!(kernels().relu(&input, &mut output).is_err());
    assert_values(&output, &[42.0]);
}

#[test]
fn scratch_limit_is_explicit_and_exact_boundary_is_accepted() {
    assert!(CpuF32KernelsV1::new(0).is_err());
    assert!(CpuF32KernelsV1::new(usize::MAX).is_err());
    let input = tensor(&[1.0, 2.0]);
    let mut output = tensor(&[42.0; 2]);
    assert!(CpuF32KernelsV1::new(7)
        .unwrap()
        .relu(&input, &mut output)
        .is_err());
    assert_values(&output, &[42.0; 2]);
    let exact = CpuF32KernelsV1::new(8).unwrap();
    assert_eq!(exact.max_scratch_bytes(), 8);
    assert_eq!(
        exact
            .relu(&input, &mut output)
            .unwrap()
            .scratch_payload_bytes,
        8
    );
    assert_values(&output, &[1.0, 2.0]);
}

#[test]
fn gather_retains_index_order_duplicates_and_value_bits() {
    let input = tensor(&[2.0, -0.0, 3.0]);
    let mut output = tensor(&[99.0; 3]);
    kernels().gather(&input, &[2, 1, 2], &mut output).unwrap();
    assert_values(&output, &[3.0, -0.0, 3.0]);
    let before = bytes(&output);
    for indices in [&[0, 3, 0][..], &[usize::MAX][..], &[][..]] {
        assert!(kernels().gather(&input, indices, &mut output).is_err());
        assert_eq!(bytes(&output), before);
    }
}

#[test]
fn scatter_add_duplicates_follow_input_order_and_preserve_other_values() {
    let source = tensor(&[16_777_216.0, 1.0, -16_777_216.0]);
    let mut output = tensor(&[0.0, -0.0, 7.0]);
    kernels()
        .scatter_add(&source, &[0, 0, 0], &mut output)
        .unwrap();
    assert_values(&output, &[0.0, -0.0, 7.0]);
    kernels()
        .scatter_add(&tensor(&[2.0, 3.0]), &[2, 0], &mut output)
        .unwrap();
    assert_values(&output, &[3.0, -0.0, 9.0]);
}

#[test]
fn scatter_errors_preserve_destination_after_partial_staging() {
    let mut output = tensor(&[42.0, f32::MAX]);
    let before = bytes(&output);
    assert!(kernels()
        .scatter_add(&tensor(&[1.0, f32::MAX]), &[0, 1], &mut output)
        .is_err());
    assert_eq!(bytes(&output), before);
    assert!(kernels()
        .scatter_add(&tensor(&[1.0, 1.0]), &[0, 2], &mut output)
        .is_err());
    assert_eq!(bytes(&output), before);
    assert!(CpuF32KernelsV1::new(4)
        .unwrap()
        .scatter_add(&tensor(&[1.0]), &[0], &mut output)
        .is_err());
    assert_eq!(bytes(&output), before);
    assert!(kernels()
        .scatter_add(&tensor(&[1.0]), &[], &mut output)
        .is_err());
    assert_eq!(bytes(&output), before);
}

#[test]
fn scatter_rejects_nonfinite_existing_destination_but_overwrite_can_replace_it() {
    let mut output = tensor(&[f32::NAN, 1.0]);
    let before = bytes(&output);
    assert!(kernels()
        .scatter_add(&tensor(&[1.0]), &[1], &mut output)
        .is_err());
    assert_eq!(bytes(&output), before);
    kernels().relu(&tensor(&[2.0, 3.0]), &mut output).unwrap();
    assert_values(&output, &[2.0, 3.0]);
}

#[test]
fn memory_copy_size_overflow_remains_rejected_without_mutation() {
    let source = tensor(&[1.0]);
    let mut destination = tensor(&[42.0]);
    let queue = CpuDevice::new().unwrap().create_queue().unwrap();
    assert!(queue
        .copy_buffer(&source, 0, &mut destination, 0, u64::MAX)
        .is_err());
    assert_values(&destination, &[42.0]);
}
