//! Tiled matrix-matrix product `C = A * B` over packed-bf16 `u16` inputs.
//!
//! Numeric policy matches the elementwise family: bf16 storage, f32 compute.
//! Both operand tiles are widened with exact bit shifts, accumulated in f32
//! explicit-FMA chains over `k` in ascending order, and narrowed once at the
//! final store with round-to-nearest-even. The CPU oracle replays that exact
//! sequence, so GPU results are bit-for-bit reproducible regardless of
//! compiler contraction settings.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const BF16_GEMM_SOURCE: &str = r#"
__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16_bits(float value) {
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        // NaN: quiet it and avoid rounding into infinity.
        bits |= 0x00400000u;
        return (unsigned short)(bits >> 16);
    }
    unsigned int lsb = (bits >> 16) & 1u;
    bits += 0x7FFFu + lsb;
    return (unsigned short)(bits >> 16);
}

extern "C" __global__ void nnis_bf16_gemm_f32acc(
    const unsigned short* matrix_a,
    const unsigned short* matrix_b,
    unsigned short* output,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k
) {
    extern __shared__ float tile[];
    const unsigned int tile_side = blockDim.x;
    float* tile_a = tile;
    float* tile_b = tile + tile_side * tile_side;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned long long row =
        (unsigned long long)blockIdx.x * blockDim.x + ty;
    const unsigned long long col =
        (unsigned long long)blockIdx.y * blockDim.y + tx;

    float value = 0.0f;
    for (unsigned long long tile_start = 0; tile_start < k;
         tile_start += tile_side) {
        const unsigned long long a_col = tile_start + tx;
        if (row < m && a_col < k) {
            tile_a[ty * tile_side + tx] =
                bf16_bits_to_f32(matrix_a[row * k + a_col]);
        } else {
            tile_a[ty * tile_side + tx] = 0.0f;
        }
        const unsigned long long b_row = tile_start + ty;
        if (b_row < k && col < n) {
            tile_b[ty * tile_side + tx] =
                bf16_bits_to_f32(matrix_b[b_row * n + col]);
        } else {
            tile_b[ty * tile_side + tx] = 0.0f;
        }
        __syncthreads();
        for (unsigned int depth = 0; depth < tile_side; ++depth) {
            value = fmaf(
                tile_a[ty * tile_side + depth],
                tile_b[depth * tile_side + tx],
                value
            );
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        output[row * n + col] = f32_to_bf16_bits(value);
    }
}

extern "C" __global__ void nnis_bf16_gemm_nt_f32acc(
    const unsigned short* matrix_a,
    const unsigned short* matrix_b,
    unsigned short* output,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k
) {
    extern __shared__ float tile[];
    const unsigned int tile_side = blockDim.x;
    float* tile_a = tile;
    float* tile_b = tile + tile_side * tile_side;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned long long row =
        (unsigned long long)blockIdx.x * blockDim.x + ty;
    const unsigned long long col =
        (unsigned long long)blockIdx.y * blockDim.y + tx;

    float value = 0.0f;
    for (unsigned long long tile_start = 0; tile_start < k;
         tile_start += tile_side) {
        const unsigned long long a_col = tile_start + tx;
        if (row < m && a_col < k) {
            tile_a[ty * tile_side + tx] =
                bf16_bits_to_f32(matrix_a[row * k + a_col]);
        } else {
            tile_a[ty * tile_side + tx] = 0.0f;
        }
        // B stored row-major (n x k): B^T[e][col] = B[col][e].
        const unsigned long long depth = tile_start + ty;
        if (col < n && depth < k) {
            tile_b[ty * tile_side + tx] =
                bf16_bits_to_f32(matrix_b[col * k + depth]);
        } else {
            tile_b[ty * tile_side + tx] = 0.0f;
        }
        __syncthreads();
        for (unsigned int d2 = 0; d2 < tile_side; ++d2) {
            value = fmaf(
                tile_a[ty * tile_side + d2],
                tile_b[d2 * tile_side + tx],
                value
            );
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        output[row * n + col] = f32_to_bf16_bits(value);
    }
}
"#;

const DEFAULT_TILE_SIDE: u32 = 16;
/// CUDA limits gridDim.y/z to 65535 blocks.
const MAX_GRID_Y_BLOCKS: u64 = 65_535;

/// Context-bound tiled packed-bf16 matrix-matrix product with f32 compute.
#[derive(Debug)]
pub struct Bf16Gemm {
    gemm: Kernel,
    gemm_nt: Kernel,
    tile_side: u32,
}

impl Bf16Gemm {
    /// Compile (or reuse cached CUBIN) and load the default GEMM kernel.
    pub fn load(context: &Arc<Context>, compiler: &JitCompiler) -> Result<Self> {
        Self::load_with_tile_side(context, compiler, DEFAULT_TILE_SIDE)
    }

    /// Load the kernel with an explicitly selected power-of-two tile side.
    ///
    /// One block computes a `tile_side x tile_side` output tile with the same
    /// block shape, so the total thread count is `tile_side^2`.
    pub fn load_with_tile_side(
        context: &Arc<Context>,
        compiler: &JitCompiler,
        tile_side: u32,
    ) -> Result<Self> {
        if tile_side == 0 || !tile_side.is_power_of_two() {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm tile side {tile_side} is not a non-zero power of two"
            )));
        }
        let shared_memory_bytes = Self::shared_memory_bytes(tile_side)?;
        let code =
            compiler.compile_cubin(BF16_GEMM_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let gemm = module.get_function("nnis_bf16_gemm_f32acc")?;
        let gemm_nt = module.get_function("nnis_bf16_gemm_nt_f32acc")?;
        let attributes = gemm.attributes()?;
        let attributes_nt = gemm_nt.attributes()?;
        let threads_per_block = u64::from(tile_side)
            .checked_mul(u64::from(tile_side))
            .ok_or_else(|| NnisError::invalid_input("bf16 gemm block size overflows"))?;
        if threads_per_block > u64::from(attributes.max_threads_per_block) {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm tile side {tile_side} implies {threads_per_block} threads per block; \
                 function limit is {}",
                attributes.max_threads_per_block
            )));
        }
        if shared_memory_bytes as usize > attributes.max_dynamic_shared_memory_bytes as usize {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm requires {shared_memory_bytes} shared-memory bytes; \
                 function limit is {}",
                attributes.max_dynamic_shared_memory_bytes
            )));
        }
        let threads_limit = u64::from(attributes_nt.max_threads_per_block);
        if threads_per_block > threads_limit {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm-nt tile side {tile_side} implies {threads_per_block} threads; \
                 function limit is {threads_limit}"
            )));
        }
        if shared_memory_bytes as usize > attributes_nt.max_dynamic_shared_memory_bytes as usize {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm-nt requires {shared_memory_bytes} shared-memory bytes; \
                 function limit is {}",
                attributes_nt.max_dynamic_shared_memory_bytes
            )));
        }
        Ok(Self {
            gemm,
            gemm_nt,
            tile_side,
        })
    }

    /// CUDA block/tile side used along both matrix dimensions.
    pub fn tile_side(&self) -> u32 {
        self.tile_side
    }

    /// Compute `output = matrix_a * matrix_b` and wait for completion.
    ///
    /// Shapes: `matrix_a` holds `m * k` row-major packed-bf16 elements,
    /// `matrix_b` holds `k * n`, and `output` receives `m * n`. Accumulation
    /// happens in f32; the result is narrowed to bf16 once per element. An
    /// empty inner dimension zeroes the output; zero rows or columns leave
    /// nothing to write.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<u16>,
        matrix_b: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result =
            unsafe { self.enqueue_gemm(stream, matrix_a, matrix_b, output, m, n, k) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Compute `output = matrix_a * matrix_b_transposed` and wait.
    ///
    /// `matrix_a` holds `m * k` row-major packed-bf16 elements; `matrix_b`
    /// holds `n * k` row-major elements so the math reads `B` transposed
    /// (the `Q * K^T` score form); `output` receives `m * n`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_transposed_b(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<u16>,
        matrix_b: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result =
            unsafe { self.enqueue_gemm_transposed_b(stream, matrix_a, matrix_b, output, m, n, k) };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the transposed-B GEMM without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_gemm_transposed_b(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<u16>,
        matrix_b: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.validate_execution_nt(stream, matrix_a, matrix_b, output, m, n, k)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            // SAFETY: the output lifetime obligation is documented above.
            return unsafe { output.zero_async(stream) };
        }
        let shared_memory_bytes = Self::shared_memory_bytes(self.tile_side)?;
        let grid_x = u32::try_from(m.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("bf16 gemm-nt exceeds u32 row blocks"))?;
        let grid_y = u64::try_from(n.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("bf16 gemm-nt column blocks exceed u64"))?;
        if grid_y > MAX_GRID_Y_BLOCKS {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm-nt requires {grid_y} column blocks; \
                 gridDim.y limit is {MAX_GRID_Y_BLOCKS}"
            )));
        }
        let (m, n, k) = (
            u64::try_from(m).map_err(|_| NnisError::invalid_input("bf16 gemm-nt m exceeds u64"))?,
            u64::try_from(n).map_err(|_| NnisError::invalid_input("bf16 gemm-nt n exceeds u64"))?,
            u64::try_from(k).map_err(|_| NnisError::invalid_input("bf16 gemm-nt k exceeds u64"))?,
        );
        let config = LaunchConfig::new(
            Dim3::new(grid_x, grid_y as u32, 1),
            Dim3::new(self.tile_side, self.tile_side, 1),
        )
        .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(6, 3);
        arguments
            .push_buffer(matrix_a)
            .push_buffer(matrix_b)
            .push_buffer(output)
            .push(m)
            .push(n)
            .push(k);
        let launch = KernelLaunch::new(&self.gemm_nt, stream, config);
        // SAFETY: argument order/widths match `nnis_bf16_gemm_nt_f32acc`;
        // the caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    /// Enqueue the GEMM without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_gemm(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<u16>,
        matrix_b: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.validate_execution(stream, matrix_a, matrix_b, output, m, n, k)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            // SAFETY: the output lifetime obligation is documented above.
            return unsafe { output.zero_async(stream) };
        }
        let shared_memory_bytes = Self::shared_memory_bytes(self.tile_side)?;
        let grid_x = u32::try_from(m.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("bf16 gemm exceeds u32::MAX row blocks"))?;
        let grid_y = u64::try_from(n.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("bf16 gemm column blocks exceed u64"))?;
        if grid_y > MAX_GRID_Y_BLOCKS {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm requires {grid_y} column blocks; \
                 gridDim.y limit is {MAX_GRID_Y_BLOCKS}"
            )));
        }
        let (m, n, k) = (
            u64::try_from(m).map_err(|_| NnisError::invalid_input("bf16 gemm m exceeds u64"))?,
            u64::try_from(n).map_err(|_| NnisError::invalid_input("bf16 gemm n exceeds u64"))?,
            u64::try_from(k).map_err(|_| NnisError::invalid_input("bf16 gemm k exceeds u64"))?,
        );
        let config = LaunchConfig::new(
            Dim3::new(grid_x, grid_y as u32, 1),
            Dim3::new(self.tile_side, self.tile_side, 1),
        )
        .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(6, 3);
        arguments
            .push_buffer(matrix_a)
            .push_buffer(matrix_b)
            .push_buffer(output)
            .push(m)
            .push(n)
            .push(k);
        let launch = KernelLaunch::new(&self.gemm, stream, config);
        // SAFETY: argument order/widths match `nnis_bf16_gemm_f32acc`; the
        // caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_execution_nt(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<u16>,
        matrix_b: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| NnisError::invalid_input("bf16 gemm-nt shape overflows usize"))?;
        let expected_b = n
            .checked_mul(k)
            .ok_or_else(|| NnisError::invalid_input("bf16 gemm-nt shape overflows usize"))?;
        let expected_output = m
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("bf16 gemm-nt shape overflows usize"))?;
        if matrix_a.len() != expected_a {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm-nt matrix_a has {} elements; shape ({m}, {k}) requires {expected_a}",
                matrix_a.len()
            )));
        }
        if matrix_b.len() != expected_b {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm-nt matrix_b has {} elements; shape ({n}, {k}) requires {expected_b}",
                matrix_b.len()
            )));
        }
        if output.len() != expected_output {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm-nt output has {} elements; shape ({m}, {n}) requires {expected_output}",
                output.len()
            )));
        }
        let context = self.gemm.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, matrix_a.ctx())
            || !Arc::ptr_eq(context, matrix_b.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "bf16 gemm-nt stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }

    fn shared_memory_bytes(tile_side: u32) -> Result<u32> {
        tile_side
            .checked_mul(tile_side)
            .and_then(|threads| threads.checked_mul(2))
            .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>() as u32))
            .ok_or_else(|| NnisError::invalid_input("bf16 gemm shared-memory size overflows"))
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_execution(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<u16>,
        matrix_b: &DeviceBuffer<u16>,
        output: &DeviceBuffer<u16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| NnisError::invalid_input("bf16 gemm shape overflows usize"))?;
        let expected_b = k
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("bf16 gemm shape overflows usize"))?;
        let expected_output = m
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("bf16 gemm shape overflows usize"))?;
        if matrix_a.len() != expected_a {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm matrix_a has {} elements; shape ({m}, {k}) requires {expected_a}",
                matrix_a.len()
            )));
        }
        if matrix_b.len() != expected_b {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm matrix_b has {} elements; shape ({k}, {n}) requires {expected_b}",
                matrix_b.len()
            )));
        }
        if output.len() != expected_output {
            return Err(NnisError::invalid_input(format!(
                "bf16 gemm output has {} elements; shape ({m}, {n}) requires {expected_output}",
                output.len()
            )));
        }
        let context = self.gemm.context();
        if !Arc::ptr_eq(context, stream.ctx())
            || !Arc::ptr_eq(context, matrix_a.ctx())
            || !Arc::ptr_eq(context, matrix_b.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "bf16 gemm stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::{bf16_bits_to_f32, f32_to_bf16_rne, gpu_context};

    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (2, 3, 4),
        (7, 31, 5),
        (16, 16, 16),
        (17, 17, 17),
        (32, 48, 64),
        (33, 65, 127),
        (128, 96, 129),
    ];

    /// Values chosen so every product and partial sum survives bf16 storage:
    /// coarse magnitudes with exact binary fractions.
    fn host_matrix_a(m: usize, k: usize) -> Vec<f32> {
        (0..m * k)
            .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
            .collect()
    }

    fn host_matrix_b(k: usize, n: usize) -> Vec<f32> {
        (0..k * n)
            .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.125)
            .collect()
    }

    fn to_bits(values: &[f32]) -> Vec<u16> {
        values.iter().copied().map(f32_to_bf16_rne).collect()
    }

    /// Replays the kernel exactly: widen both operands from stored bf16,
    /// accumulate one explicit-FMA chain over `k` ascending, narrow RNE once.
    fn reference_gemm(a_bits: &[u16], b_bits: &[u16], m: usize, n: usize, k: usize) -> Vec<u16> {
        let mut output = vec![0_u16; m * n];
        for row in 0..m {
            for col in 0..n {
                let value = (0..k).fold(0.0_f32, |value, depth| {
                    bf16_bits_to_f32(a_bits[row * k + depth])
                        .mul_add(bf16_bits_to_f32(b_bits[depth * n + col]), value)
                });
                output[row * n + col] = f32_to_bf16_rne(value);
            }
        }
        output
    }

    #[test]
    fn bf16_gemm_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = Bf16Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(m, n, k) in SHAPES {
            let a_bits = to_bits(&host_matrix_a(m, k));
            let b_bits = to_bits(&host_matrix_b(k, n));
            let matrix_a = DeviceBuffer::from_host(&context, &stream, &a_bits).unwrap();
            let matrix_b = DeviceBuffer::from_host(&context, &stream, &b_bits).unwrap();
            // Pre-fill the output so a skipped kernel cannot pass silently.
            let output_host = vec![0xFFFF_u16; m * n];
            let output = DeviceBuffer::from_host(&context, &stream, &output_host).unwrap();

            gemm.gemm(&stream, &matrix_a, &matrix_b, &output, m, n, k)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let expected = reference_gemm(&a_bits, &b_bits, m, n, k);
            for index in 0..m * n {
                assert_eq!(
                    actual[index], expected[index],
                    "ordered bf16 gemm mismatch at flat {index} shape ({m}, {n}, {k}): \
                     bits {:04x} != {:04x}",
                    actual[index], expected[index]
                );
            }
        }
    }

    #[test]
    fn bf16_gemm_zero_inner_dimension_zeroes_output_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = Bf16Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        // Shapes (m=2, k=0) and (k=0, n=3): both operand buffers are empty.
        let matrix_a = DeviceBuffer::<u16>::new(&context, 0).unwrap();
        let matrix_b = DeviceBuffer::<u16>::new(&context, 0).unwrap();
        let output_host = vec![0x3F80_u16; 6];
        let output = DeviceBuffer::from_host(&context, &stream, &output_host).unwrap();

        gemm.gemm(&stream, &matrix_a, &matrix_b, &output, 2, 3, 0)
            .unwrap();
        let output_host = output.to_vec(&stream).unwrap();
        assert!(output_host.iter().all(|&value| value == 0));
    }

    #[test]
    fn bf16_gemm_rejects_invalid_shapes_and_tiles_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(Bf16Gemm::load_with_tile_side(&context, &compiler, 0).is_err());
        assert!(Bf16Gemm::load_with_tile_side(&context, &compiler, 24).is_err());
        assert!(Bf16Gemm::load_with_tile_side(&context, &compiler, 64).is_err());

        let gemm = Bf16Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let matrix_a = DeviceBuffer::<u16>::new(&context, 12).unwrap(); // 3 x 4
        let short_matrix_b = DeviceBuffer::<u16>::new(&context, 11).unwrap(); // needs 12
        let output = DeviceBuffer::<u16>::new(&context, 9).unwrap(); // 3 x 3
        let error = gemm
            .gemm(&stream, &matrix_a, &short_matrix_b, &output, 3, 3, 4)
            .unwrap_err();
        assert!(error.to_string().contains("requires 12"), "{error}");

        let matrix_b = DeviceBuffer::<u16>::new(&context, 12).unwrap();
        let long_output = DeviceBuffer::<u16>::new(&context, 10).unwrap();
        let error = gemm
            .gemm(&stream, &matrix_a, &matrix_b, &long_output, 3, 3, 4)
            .unwrap_err();
        assert!(
            error.to_string().contains("shape (3, 3) requires 9"),
            "{error}"
        );
    }
    #[test]
    fn bf16_gemm_transposed_b_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = Bf16Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(m, n, k) in SHAPES {
            let a_bits = to_bits(&host_matrix_a(m, k));
            // B stored row-major as (n x k).
            let b_bits = to_bits(&host_matrix_a(n, k));
            let matrix_a = DeviceBuffer::from_host(&context, &stream, &a_bits).unwrap();
            let matrix_b = DeviceBuffer::from_host(&context, &stream, &b_bits).unwrap();
            let output_host = vec![0xFFFF_u16; m * n];
            let output = DeviceBuffer::from_host(&context, &stream, &output_host).unwrap();

            gemm.gemm_transposed_b(&stream, &matrix_a, &matrix_b, &output, m, n, k)
                .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            for row in 0..m {
                for col in 0..n {
                    let expected = (0..k).fold(0.0_f32, |value, depth| {
                        bf16_bits_to_f32(a_bits[row * k + depth])
                            .mul_add(bf16_bits_to_f32(b_bits[col * k + depth]), value)
                    });
                    assert_eq!(
                        actual[row * n + col],
                        f32_to_bf16_rne(expected),
                        "ordered bf16 gemm-nt mismatch at ({row}, {col}) shape ({m}, {n}, {k})"
                    );
                }
            }
        }
    }

    #[test]
    fn bf16_gemm_transposed_b_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = Bf16Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let matrix_a = DeviceBuffer::<u16>::new(&context, 12).unwrap(); // 3 x 4
        let short_matrix_b = DeviceBuffer::<u16>::new(&context, 11).unwrap(); // needs 12
        let output = DeviceBuffer::<u16>::new(&context, 9).unwrap(); // 3 x 3
        let error = gemm
            .gemm_transposed_b(&stream, &matrix_a, &short_matrix_b, &output, 3, 3, 4)
            .unwrap_err();
        assert!(
            error.to_string().contains("shape (3, 4) requires 12"),
            "{error}"
        );

        let matrix_b = DeviceBuffer::<u16>::new(&context, 12).unwrap();
        let long_output = DeviceBuffer::<u16>::new(&context, 10).unwrap();
        let error = gemm
            .gemm_transposed_b(&stream, &matrix_a, &matrix_b, &long_output, 3, 3, 4)
            .unwrap_err();
        assert!(
            error.to_string().contains("shape (3, 3) requires 9"),
            "{error}"
        );
    }
}
