//! Tiled matrix-matrix product `C = A * B` for row-major `f32` matrices.
//!
//! One thread block computes one `tile x tile` output tile. A and B tiles are
//! staged cooperatively in dynamic shared memory; every output element is an
//! explicit-FMA accumulation over `k` in ascending order, so the GPU result is
//! bit-for-bit reproducible against the CPU oracle regardless of compiler
//! contraction settings.

use nnis_jit::{
    CompileOptions, Dim3, JitCompiler, Kernel, KernelArgs, KernelLaunch, LaunchConfig, Module,
};
use nnis_rt::{Context, DeviceBuffer, NnisError, Result, Stream};
use std::sync::Arc;

const GEMM_SOURCE: &str = r#"
extern "C" __global__ void nnis_gemm_f32(
    const float* matrix_a,
    const float* matrix_b,
    float* output,
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
            tile_a[ty * tile_side + tx] = matrix_a[row * k + a_col];
        } else {
            tile_a[ty * tile_side + tx] = 0.0f;
        }
        const unsigned long long b_row = tile_start + ty;
        if (b_row < k && col < n) {
            tile_b[ty * tile_side + tx] = matrix_b[b_row * n + col];
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
        output[row * n + col] = value;
    }
}

extern "C" __global__ void nnis_gemm_nt_f32(
    const float* matrix_a,
    const float* matrix_b,
    float* output,
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
            tile_a[ty * tile_side + tx] = matrix_a[row * k + a_col];
        } else {
            tile_a[ty * tile_side + tx] = 0.0f;
        }
        // B is stored row-major as (n x k): B^T[e][col] = B[col][e].
        const unsigned long long depth = tile_start + ty;
        if (col < n && depth < k) {
            tile_b[ty * tile_side + tx] = matrix_b[col * k + depth];
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
        output[row * n + col] = value;
    }
}

extern "C" __global__ void nnis_gemm_batched_f32(
    const float* matrix_a,
    const float* matrix_b,
    float* output,
    unsigned int batches,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k
) {
    extern __shared__ float tile[];
    const unsigned int tile_side = blockDim.x;
    float* tile_a = tile;
    float* tile_b = tile + tile_side * tile_side;

    // One gridDim.z layer owns one packed [batches][m][n] product.
    const unsigned long long batch = blockIdx.z;
    matrix_a += batch * m * k;
    matrix_b += batch * k * n;
    output += batch * m * n;

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
            tile_a[ty * tile_side + tx] = matrix_a[row * k + a_col];
        } else {
            tile_a[ty * tile_side + tx] = 0.0f;
        }
        const unsigned long long b_row = tile_start + ty;
        if (b_row < k && col < n) {
            tile_b[ty * tile_side + tx] = matrix_b[b_row * n + col];
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
        output[row * n + col] = value;
    }
}

extern "C" __global__ void nnis_gemm_nt_batched_f32(
    const float* matrix_a,
    const float* matrix_b,
    float* output,
    unsigned int batches,
    unsigned long long m,
    unsigned long long n,
    unsigned long long k
) {
    extern __shared__ float tile[];
    const unsigned int tile_side = blockDim.x;
    float* tile_a = tile;
    float* tile_b = tile + tile_side * tile_side;

    // One gridDim.z layer owns one packed [batches][m][n] product over
    // [batches][n][k] transposed-B operands.
    const unsigned long long batch = blockIdx.z;
    matrix_a += batch * m * k;
    matrix_b += batch * n * k;
    output += batch * m * n;

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
            tile_a[ty * tile_side + tx] = matrix_a[row * k + a_col];
        } else {
            tile_a[ty * tile_side + tx] = 0.0f;
        }
        // B is stored row-major as (n x k): B^T[e][col] = B[col][e].
        const unsigned long long depth = tile_start + ty;
        if (col < n && depth < k) {
            tile_b[ty * tile_side + tx] = matrix_b[col * k + depth];
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
        output[row * n + col] = value;
    }
}
"#;

const DEFAULT_TILE_SIDE: u32 = 16;
/// CUDA limits gridDim.y/z to 65535 blocks.
const MAX_GRID_Y_BLOCKS: u64 = 65_535;

/// Context-bound tiled `f32` matrix-matrix product.
#[derive(Debug)]
pub struct F32Gemm {
    gemm: Kernel,
    gemm_nt: Kernel,
    gemm_batched: Kernel,
    gemm_nt_batched: Kernel,
    tile_side: u32,
}

impl F32Gemm {
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
                "gemm tile side {tile_side} is not a non-zero power of two"
            )));
        }
        let shared_memory_bytes = Self::shared_memory_bytes(tile_side)?;
        let code = compiler.compile_cubin(GEMM_SOURCE, &CompileOptions::for_device(context))?;
        let module = Module::load(context, &code)?;
        let gemm = module.get_function("nnis_gemm_f32")?;
        let gemm_nt = module.get_function("nnis_gemm_nt_f32")?;
        let gemm_batched = module.get_function("nnis_gemm_batched_f32")?;
        let gemm_nt_batched = module.get_function("nnis_gemm_nt_batched_f32")?;
        let attributes = gemm.attributes()?;
        let attributes_nt = gemm_nt.attributes()?;
        let threads_per_block = u64::from(tile_side)
            .checked_mul(u64::from(tile_side))
            .ok_or_else(|| NnisError::invalid_input("gemm block size overflows"))?;
        if threads_per_block > u64::from(attributes.max_threads_per_block) {
            return Err(NnisError::invalid_input(format!(
                "gemm tile side {tile_side} implies {threads_per_block} threads per block; \
                 function limit is {}",
                attributes.max_threads_per_block
            )));
        }
        if shared_memory_bytes as usize > attributes.max_dynamic_shared_memory_bytes as usize {
            return Err(NnisError::invalid_input(format!(
                "gemm requires {shared_memory_bytes} shared-memory bytes; function limit is {}",
                attributes.max_dynamic_shared_memory_bytes
            )));
        }
        let threads_limit = u64::from(attributes.max_threads_per_block);
        if threads_per_block > threads_limit {
            return Err(NnisError::invalid_input(format!(
                "gemm-nt tile side {tile_side} implies {threads_per_block} threads per block; \
                 function limit is {threads_limit}"
            )));
        }
        if shared_memory_bytes as usize > attributes_nt.max_dynamic_shared_memory_bytes as usize {
            return Err(NnisError::invalid_input(format!(
                "gemm-nt requires {shared_memory_bytes} shared-memory bytes; function limit is {}",
                attributes_nt.max_dynamic_shared_memory_bytes
            )));
        }
        for name in ["gemm-batched", "gemm-nt-batched"] {
            let function = if name == "gemm-batched" {
                &gemm_batched
            } else {
                &gemm_nt_batched
            };
            let batched_attributes = function.attributes()?;
            if threads_per_block > u64::from(batched_attributes.max_threads_per_block) {
                return Err(NnisError::invalid_input(format!(
                    "{name} tile side {tile_side} implies {threads_per_block} threads per block; \
                     function limit is {}",
                    batched_attributes.max_threads_per_block
                )));
            }
            if shared_memory_bytes as usize
                > batched_attributes.max_dynamic_shared_memory_bytes as usize
            {
                return Err(NnisError::invalid_input(format!(
                    "{name} requires {shared_memory_bytes} shared-memory bytes; \
                     function limit is {}",
                    batched_attributes.max_dynamic_shared_memory_bytes
                )));
            }
        }
        Ok(Self {
            gemm,
            gemm_nt,
            gemm_batched,
            gemm_nt_batched,
            tile_side,
        })
    }

    /// CUDA block/tile side used along both matrix dimensions.
    pub fn tile_side(&self) -> u32 {
        self.tile_side
    }

    /// Compute `output = matrix_a * matrix_b` and wait for completion.
    ///
    /// Shapes: `matrix_a` holds `m * k` row-major elements, `matrix_b` holds
    /// `k * n`, and `output` receives `m * n`. An empty inner dimension zeroes
    /// the output; zero rows or columns leave nothing to write.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
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

    /// Compute `output = matrix_a * matrix_b_transposed` and wait for
    /// completion.
    ///
    /// Shapes: `matrix_a` holds `m * k` row-major elements, `matrix_b` holds
    /// `n * k` row-major elements (so the math reads `B` transposed), and
    /// `output` receives `m * n`. This is the score form `Q * K^T` used by
    /// attention when both operands store rows per token. Empty inner
    /// dimension zeroes the output; zero rows or columns leave nothing to
    /// write.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_transposed_b(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
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
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
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
            .map_err(|_| NnisError::invalid_input("gemm-nt exceeds u32::MAX row blocks"))?;
        let grid_y = u64::try_from(n.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("gemm-nt column blocks exceed u64"))?;
        if grid_y > MAX_GRID_Y_BLOCKS {
            return Err(NnisError::invalid_input(format!(
                "gemm-nt requires {grid_y} column blocks; \
                 gridDim.y limit is {MAX_GRID_Y_BLOCKS}"
            )));
        }
        let (m, n, k) = (
            u64::try_from(m).map_err(|_| NnisError::invalid_input("gemm-nt m exceeds u64"))?,
            u64::try_from(n).map_err(|_| NnisError::invalid_input("gemm-nt n exceeds u64"))?,
            u64::try_from(k).map_err(|_| NnisError::invalid_input("gemm-nt k exceeds u64"))?,
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
        // SAFETY: argument order/widths match `nnis_gemm_nt_f32`; the caller
        // owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    /// Compute one transposed-B product per packed batch and wait.
    ///
    /// Layout: `matrix_a` holds `batches * m * k`, `matrix_b` holds
    /// `batches * n * k`, and `output` receives `batches * m * n`, each
    /// batch contiguous. One launch covers every batch via gridDim.z; the
    /// per-batch trajectory is identical to [`Self::gemm_transposed_b`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_transposed_b_batched(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        batches: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe {
            self.enqueue_gemm_transposed_b_batched(
                stream, matrix_a, matrix_b, output, batches, m, n, k,
            )
        };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the batched transposed-B GEMM without synchronizing the
    /// stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_gemm_transposed_b_batched(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        batches: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let grid_z = Self::validate_batched_execution(
            stream, matrix_a, matrix_b, output, batches, m, n, k, true,
        )?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            // SAFETY: the output lifetime obligation is documented above.
            return unsafe { output.zero_async(stream) };
        }
        let shared_memory_bytes = Self::shared_memory_bytes(self.tile_side)?;
        let grid_x = u32::try_from(m.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("batched gemm-nt exceeds u32 row blocks"))?;
        let grid_y = u64::try_from(n.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("batched gemm-nt column blocks exceed u64"))?;
        if grid_y > MAX_GRID_Y_BLOCKS {
            return Err(NnisError::invalid_input(format!(
                "batched gemm-nt requires {grid_y} column blocks; \
                 gridDim.y limit is {MAX_GRID_Y_BLOCKS}"
            )));
        }
        let (m, n, k) = (
            u64::try_from(m)
                .map_err(|_| NnisError::invalid_input("batched gemm-nt m exceeds u64"))?,
            u64::try_from(n)
                .map_err(|_| NnisError::invalid_input("batched gemm-nt n exceeds u64"))?,
            u64::try_from(k)
                .map_err(|_| NnisError::invalid_input("batched gemm-nt k exceeds u64"))?,
        );
        let config = LaunchConfig::new(
            Dim3::new(grid_x, grid_y as u32, grid_z),
            Dim3::new(self.tile_side, self.tile_side, 1),
        )
        .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(7, 4);
        arguments
            .push_buffer(matrix_a)
            .push_buffer(matrix_b)
            .push_buffer(output)
            .push(u32::try_from(batches).expect("validated within gridDim.z limit"))
            .push(m)
            .push(n)
            .push(k);
        let launch = KernelLaunch::new(&self.gemm_nt_batched, stream, config);
        // SAFETY: argument order/widths match `nnis_gemm_nt_batched_f32`;
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
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
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
            .map_err(|_| NnisError::invalid_input("gemm exceeds u32::MAX row blocks"))?;
        let grid_y = u64::try_from(n.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("gemm column blocks exceed u64"))?;
        if grid_y > MAX_GRID_Y_BLOCKS {
            return Err(NnisError::invalid_input(format!(
                "gemm requires {grid_y} column blocks; gridDim.y limit is {MAX_GRID_Y_BLOCKS}"
            )));
        }
        let (m, n, k) = (
            u64::try_from(m).map_err(|_| NnisError::invalid_input("gemm m exceeds u64::MAX"))?,
            u64::try_from(n).map_err(|_| NnisError::invalid_input("gemm n exceeds u64::MAX"))?,
            u64::try_from(k).map_err(|_| NnisError::invalid_input("gemm k exceeds u64::MAX"))?,
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
        // SAFETY: argument order/widths match `nnis_gemm_f32`; the caller
        // owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    /// Compute one product per packed batch and wait.
    ///
    /// Layout: `matrix_a` holds `batches * m * k`, `matrix_b` holds
    /// `batches * k * n`, and `output` receives `batches * m * n`, each
    /// batch contiguous. One launch covers every batch via gridDim.z; the
    /// per-batch trajectory is identical to [`Self::gemm`].
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_batched(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        batches: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // SAFETY: all borrows remain live until synchronization below.
        let enqueue_result = unsafe {
            self.enqueue_gemm_batched(stream, matrix_a, matrix_b, output, batches, m, n, k)
        };
        match enqueue_result {
            Ok(()) => stream.synchronize(),
            Err(error) => {
                let _ = stream.synchronize();
                Err(error)
            }
        }
    }

    /// Enqueue the batched GEMM without synchronizing the stream.
    ///
    /// # Safety
    ///
    /// All buffers, the stream, and this kernel must remain alive and
    /// otherwise untouched until the stream completes.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_gemm_batched(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        batches: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let grid_z = Self::validate_batched_execution(
            stream, matrix_a, matrix_b, output, batches, m, n, k, false,
        )?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        if k == 0 {
            // SAFETY: the output lifetime obligation is documented above.
            return unsafe { output.zero_async(stream) };
        }
        let shared_memory_bytes = Self::shared_memory_bytes(self.tile_side)?;
        let grid_x = u32::try_from(m.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("batched gemm exceeds u32 row blocks"))?;
        let grid_y = u64::try_from(n.div_ceil(self.tile_side as usize))
            .map_err(|_| NnisError::invalid_input("batched gemm column blocks exceed u64"))?;
        if grid_y > MAX_GRID_Y_BLOCKS {
            return Err(NnisError::invalid_input(format!(
                "batched gemm requires {grid_y} column blocks; \
                 gridDim.y limit is {MAX_GRID_Y_BLOCKS}"
            )));
        }
        let (m, n, k) = (
            u64::try_from(m).map_err(|_| NnisError::invalid_input("batched gemm m exceeds u64"))?,
            u64::try_from(n).map_err(|_| NnisError::invalid_input("batched gemm n exceeds u64"))?,
            u64::try_from(k).map_err(|_| NnisError::invalid_input("batched gemm k exceeds u64"))?,
        );
        let config = LaunchConfig::new(
            Dim3::new(grid_x, grid_y as u32, grid_z),
            Dim3::new(self.tile_side, self.tile_side, 1),
        )
        .with_dynamic_shared_memory(shared_memory_bytes);
        let mut arguments = KernelArgs::with_capacity(7, 4);
        arguments
            .push_buffer(matrix_a)
            .push_buffer(matrix_b)
            .push_buffer(output)
            .push(u32::try_from(batches).expect("validated within gridDim.z limit"))
            .push(m)
            .push(n)
            .push(k);
        let launch = KernelLaunch::new(&self.gemm_batched, stream, config);
        // SAFETY: argument order/widths match `nnis_gemm_batched_f32`; the
        // caller owns the asynchronous lifetime obligation.
        unsafe { launch.launch(&mut arguments) }
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_execution_nt(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| NnisError::invalid_input("gemm-nt shape overflows usize"))?;
        let expected_b = n
            .checked_mul(k)
            .ok_or_else(|| NnisError::invalid_input("gemm-nt shape overflows usize"))?;
        let expected_output = m
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("gemm-nt shape overflows usize"))?;
        if matrix_a.len() != expected_a {
            return Err(NnisError::invalid_input(format!(
                "gemm-nt matrix_a has {} elements; shape ({m}, {k}) requires {expected_a}",
                matrix_a.len()
            )));
        }
        if matrix_b.len() != expected_b {
            return Err(NnisError::invalid_input(format!(
                "gemm-nt matrix_b has {} elements; shape ({n}, {k}) requires {expected_b}",
                matrix_b.len()
            )));
        }
        if output.len() != expected_output {
            return Err(NnisError::invalid_input(format!(
                "gemm-nt output has {} elements; shape ({m}, {n}) requires {expected_output}",
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
                "gemm-nt stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }

    /// Shared batched-shape validation; returns the validated gridDim.z
    /// block count. `transposed_b` selects whether each `matrix_b` batch is
    /// stored `(n, k)` (score form) or `(k, n)`.
    #[allow(clippy::too_many_arguments)]
    fn validate_batched_execution(
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        batches: usize,
        m: usize,
        n: usize,
        k: usize,
        transposed_b: bool,
    ) -> Result<u32> {
        if batches == 0 {
            return Err(NnisError::invalid_input("gemm requires at least one batch"));
        }
        if batches > MAX_GRID_Y_BLOCKS as usize {
            return Err(NnisError::invalid_input(format!(
                "batched gemm requires {batches} batch blocks; \
                 gridDim.z limit is {MAX_GRID_Y_BLOCKS}"
            )));
        }
        let expected_a = batches
            .checked_mul(m)
            .and_then(|elements| elements.checked_mul(k))
            .ok_or_else(|| NnisError::invalid_input("batched gemm shape overflows usize"))?;
        let expected_b = if transposed_b {
            batches
                .checked_mul(n)
                .and_then(|elements| elements.checked_mul(k))
        } else {
            batches
                .checked_mul(k)
                .and_then(|elements| elements.checked_mul(n))
        }
        .ok_or_else(|| NnisError::invalid_input("batched gemm shape overflows usize"))?;
        let expected_output = batches
            .checked_mul(m)
            .and_then(|elements| elements.checked_mul(n))
            .ok_or_else(|| NnisError::invalid_input("batched gemm shape overflows usize"))?;
        if matrix_a.len() != expected_a {
            return Err(NnisError::invalid_input(format!(
                "batched gemm matrix_a has {} elements; {batches} batches of shape ({m}, {k}) \
                 requires {expected_a}",
                matrix_a.len()
            )));
        }
        if matrix_b.len() != expected_b {
            return Err(NnisError::invalid_input(format!(
                "batched gemm matrix_b has {} elements; {batches} batches require {expected_b}",
                matrix_b.len()
            )));
        }
        if output.len() != expected_output {
            return Err(NnisError::invalid_input(format!(
                "batched gemm output has {} elements; {batches} batches of shape ({m}, {n}) \
                 requires {expected_output}",
                output.len()
            )));
        }
        let context = stream.ctx();
        if !Arc::ptr_eq(context, matrix_a.ctx())
            || !Arc::ptr_eq(context, matrix_b.ctx())
            || !Arc::ptr_eq(context, output.ctx())
        {
            return Err(NnisError::invalid_input(
                "batched gemm stream and buffers must share one context",
            ));
        }
        Ok(batches as u32)
    }

    fn shared_memory_bytes(tile_side: u32) -> Result<u32> {
        tile_side
            .checked_mul(tile_side)
            .and_then(|threads| threads.checked_mul(2))
            .and_then(|floats| floats.checked_mul(std::mem::size_of::<f32>() as u32))
            .ok_or_else(|| NnisError::invalid_input("gemm shared-memory size overflows"))
    }

    #[allow(clippy::too_many_arguments)]
    fn validate_execution(
        &self,
        stream: &Stream,
        matrix_a: &DeviceBuffer<f32>,
        matrix_b: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| NnisError::invalid_input("gemm shape overflows usize"))?;
        let expected_b = k
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("gemm shape overflows usize"))?;
        let expected_output = m
            .checked_mul(n)
            .ok_or_else(|| NnisError::invalid_input("gemm shape overflows usize"))?;
        if matrix_a.len() != expected_a {
            return Err(NnisError::invalid_input(format!(
                "gemm matrix_a has {} elements; shape ({m}, {k}) requires {expected_a}",
                matrix_a.len()
            )));
        }
        if matrix_b.len() != expected_b {
            return Err(NnisError::invalid_input(format!(
                "gemm matrix_b has {} elements; shape ({k}, {n}) requires {expected_b}",
                matrix_b.len()
            )));
        }
        if output.len() != expected_output {
            return Err(NnisError::invalid_input(format!(
                "gemm output has {} elements; shape ({m}, {n}) requires {expected_output}",
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
                "gemm stream, buffers, and kernel must share one context",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nnis_rt::gpu_context;

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

    fn host_matrix_a(m: usize, k: usize) -> Vec<f32> {
        (0..m * k)
            .map(|index| (((index * 13 % 97) as f32 - 48.0) * 0.0625) + ((index % 5) as f32 - 2.0))
            .collect()
    }

    /// Per-batch variant with distinct seeds so cross-batch mixing cannot
    /// pass silently.
    fn host_batch_a(batch: usize, m: usize, k: usize) -> Vec<f32> {
        (0..m * k)
            .map(|index| {
                (((index * 13 % 97) as f32 - 48.0) * 0.0625)
                    + (((index + batch * 37) % 5) as f32 - 2.0)
            })
            .collect()
    }

    fn host_batch_b(batch: usize, rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|index| ((index * 29 % 61) as f32 - 30.0) * (0.125 + batch as f32 * 0.03125))
            .collect()
    }

    fn host_matrix_b(k: usize, n: usize) -> Vec<f32> {
        (0..k * n)
            .map(|index| ((index * 29 % 61) as f32 - 30.0) * 0.125)
            .collect()
    }

    /// Replays the kernel's evaluation order exactly: an explicit-FMA chain
    /// over `k` in ascending order for every output element.
    fn reference_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut output = vec![0.0_f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let value = (0..k).fold(0.0_f32, |value, depth| {
                    a[row * k + depth].mul_add(b[depth * n + col], value)
                });
                output[row * n + col] = value;
            }
        }
        output
    }

    #[test]
    fn gemm_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(m, n, k) in SHAPES {
            let a_host = host_matrix_a(m, k);
            let b_host = host_matrix_b(k, n);
            let matrix_a = DeviceBuffer::from_host(&context, &stream, &a_host).unwrap();
            let matrix_b = DeviceBuffer::from_host(&context, &stream, &b_host).unwrap();
            // Pre-fill the output so a skipped kernel cannot pass silently.
            let output_host = vec![f32::NAN; m * n];
            let output = DeviceBuffer::from_host(&context, &stream, &output_host).unwrap();

            gemm.gemm(&stream, &matrix_a, &matrix_b, &output, m, n, k)
                .unwrap();
            let output_host = output.to_vec(&stream).unwrap();
            let expected = reference_gemm(&a_host, &b_host, m, n, k);
            for index in 0..m * n {
                assert_eq!(
                    output_host[index].to_bits(),
                    expected[index].to_bits(),
                    "ordered gemm mismatch at flat {index} shape ({m}, {n}, {k}): \
                     {} != {}",
                    output_host[index],
                    expected[index]
                );
            }

            // Independent f64 check inside an explicit tolerance.
            for row in 0..m {
                for col in 0..n {
                    let high_precision: f64 = (0..k)
                        .map(|depth| {
                            f64::from(a_host[row * k + depth]) * f64::from(b_host[depth * n + col])
                        })
                        .sum();
                    let actual = f64::from(output_host[row * n + col]);
                    assert!(
                        (actual - high_precision).abs()
                            <= 1.0e-3_f64.max(high_precision.abs() * 1.0e-5),
                        "gemm f64 mismatch at ({row}, {col}): {actual} vs {high_precision}"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_zero_inner_dimension_zeroes_output_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        // Shapes (m=2, k=0) and (k=0, n=3): both operand buffers are empty.
        let matrix_a = DeviceBuffer::<f32>::new(&context, 0).unwrap();
        let matrix_b = DeviceBuffer::<f32>::new(&context, 0).unwrap();
        let output_host = vec![1.5_f32; 6];
        let output = DeviceBuffer::from_host(&context, &stream, &output_host).unwrap();

        gemm.gemm(&stream, &matrix_a, &matrix_b, &output, 2, 3, 0)
            .unwrap();
        let output_host = output.to_vec(&stream).unwrap();
        assert!(output_host.iter().all(|&value| value == 0.0));
    }

    #[test]
    fn gemm_rejects_invalid_shapes_and_tiles_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        assert!(F32Gemm::load_with_tile_side(&context, &compiler, 0).is_err());
        assert!(F32Gemm::load_with_tile_side(&context, &compiler, 24).is_err());
        assert!(F32Gemm::load_with_tile_side(&context, &compiler, 64).is_err());

        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let matrix_a = DeviceBuffer::<f32>::new(&context, 12).unwrap(); // 3 x 4
        let short_matrix_b = DeviceBuffer::<f32>::new(&context, 11).unwrap(); // needs 12
        let output = DeviceBuffer::<f32>::new(&context, 9).unwrap(); // 3 x 3
        let error = gemm
            .gemm(&stream, &matrix_a, &short_matrix_b, &output, 3, 3, 4)
            .unwrap_err();
        assert!(error.to_string().contains("requires 12"), "{error}");

        let matrix_b = DeviceBuffer::<f32>::new(&context, 12).unwrap();
        let long_output = DeviceBuffer::<f32>::new(&context, 10).unwrap();
        let error = gemm
            .gemm(&stream, &matrix_a, &matrix_b, &long_output, 3, 3, 4)
            .unwrap_err();
        assert!(
            error.to_string().contains("shape (3, 3) requires 9"),
            "{error}"
        );
    }

    #[test]
    fn gemm_transposed_b_matches_ordered_cpu_oracle_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        for &(m, n, k) in SHAPES {
            let a_host = host_matrix_a(m, k);
            // B stored row-major as (n x k).
            let b_host = host_matrix_a(n, k);
            let matrix_a = DeviceBuffer::from_host(&context, &stream, &a_host).unwrap();
            let matrix_b = DeviceBuffer::from_host(&context, &stream, &b_host).unwrap();
            // Pre-fill the output so a skipped kernel cannot pass silently.
            let output_host = vec![f32::NAN; m * n];
            let output = DeviceBuffer::from_host(&context, &stream, &output_host).unwrap();

            gemm.gemm_transposed_b(&stream, &matrix_a, &matrix_b, &output, m, n, k)
                .unwrap();
            let output_host = output.to_vec(&stream).unwrap();
            for row in 0..m {
                for col in 0..n {
                    // Replays the kernel: explicit-FMA chain over k ascending
                    // with the B operand read from its (col, depth) storage.
                    let expected = (0..k).fold(0.0_f32, |value, depth| {
                        a_host[row * k + depth].mul_add(b_host[col * k + depth], value)
                    });
                    assert_eq!(
                        output_host[row * n + col].to_bits(),
                        expected.to_bits(),
                        "ordered gemm-nt mismatch at ({row}, {col}) shape ({m}, {n}, {k})"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_transposed_b_rejects_invalid_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();
        let matrix_a = DeviceBuffer::<f32>::new(&context, 12).unwrap(); // 3 x 4
        let short_matrix_b = DeviceBuffer::<f32>::new(&context, 11).unwrap(); // needs 12
        let output = DeviceBuffer::<f32>::new(&context, 9).unwrap(); // 3 x 3
        let error = gemm
            .gemm_transposed_b(&stream, &matrix_a, &short_matrix_b, &output, 3, 3, 4)
            .unwrap_err();
        assert!(
            error.to_string().contains("shape (3, 4) requires 12"),
            "{error}"
        );

        let matrix_b = DeviceBuffer::<f32>::new(&context, 12).unwrap();
        let long_output = DeviceBuffer::<f32>::new(&context, 10).unwrap();
        let error = gemm
            .gemm_transposed_b(&stream, &matrix_a, &matrix_b, &long_output, 3, 3, 4)
            .unwrap_err();
        assert!(
            error.to_string().contains("shape (3, 3) requires 9"),
            "{error}"
        );
    }

    #[test]
    fn gemm_batched_bit_matches_per_batch_on_gpu() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        const BATCHED_SHAPES: &[(usize, usize, usize, usize)] =
            &[(1, 1, 1, 1), (2, 3, 4, 5), (3, 17, 31, 33), (2, 32, 48, 65)];

        for &(batches, m, n, k) in BATCHED_SHAPES {
            let mut a_host = Vec::with_capacity(batches * m * k);
            let mut b_host = Vec::with_capacity(batches * k * n);
            let mut bt_host = Vec::with_capacity(batches * n * k);
            for batch in 0..batches {
                a_host.extend(host_batch_a(batch, m, k));
                b_host.extend(host_batch_b(batch, k, n));
                bt_host.extend(host_batch_b(batch + 11, n, k));
            }
            let matrix_a = DeviceBuffer::from_host(&context, &stream, &a_host).unwrap();
            let matrix_b = DeviceBuffer::from_host(&context, &stream, &b_host).unwrap();
            let matrix_bt = DeviceBuffer::from_host(&context, &stream, &bt_host).unwrap();
            let output =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; batches * m * n])
                    .unwrap();
            let output_nt =
                DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; batches * m * n])
                    .unwrap();

            gemm.gemm_batched(&stream, &matrix_a, &matrix_b, &output, batches, m, n, k)
                .unwrap();
            gemm.gemm_transposed_b_batched(
                &stream, &matrix_a, &matrix_bt, &output_nt, batches, m, n, k,
            )
            .unwrap();
            let actual = output.to_vec(&stream).unwrap();
            let actual_nt = output_nt.to_vec(&stream).unwrap();

            for batch in 0..batches {
                // Per-batch reference through the validated single kernels.
                let head_a =
                    DeviceBuffer::from_host(&context, &stream, &host_batch_a(batch, m, k)).unwrap();
                let head_b =
                    DeviceBuffer::from_host(&context, &stream, &host_batch_b(batch, k, n)).unwrap();
                let head_bt =
                    DeviceBuffer::from_host(&context, &stream, &host_batch_b(batch + 11, n, k))
                        .unwrap();
                let head_out =
                    DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; m * n]).unwrap();
                gemm.gemm(&stream, &head_a, &head_b, &head_out, m, n, k)
                    .unwrap();
                let head_actual = head_out.to_vec(&stream).unwrap();

                let head_out_nt =
                    DeviceBuffer::from_host(&context, &stream, &vec![f32::NAN; m * n]).unwrap();
                gemm.gemm_transposed_b(&stream, &head_a, &head_bt, &head_out_nt, m, n, k)
                    .unwrap();
                let head_actual_nt = head_out_nt.to_vec(&stream).unwrap();

                let range = batch * m * n..(batch + 1) * m * n;
                for index in 0..m * n {
                    assert_eq!(
                        actual[range.start + index].to_bits(),
                        head_actual[index].to_bits(),
                        "batched plain {batches}b mismatch at batch {batch} element {index} \
                         shape ({m},{n},{k})"
                    );
                    assert_eq!(
                        actual_nt[range.start + index].to_bits(),
                        head_actual_nt[index].to_bits(),
                        "batched nt {batches}b mismatch at batch {batch} element {index} \
                         shape ({m},{n},{k})"
                    );
                }
            }
        }
    }

    #[test]
    fn gemm_batched_rejects_invalid_batches_and_shapes_before_launch() {
        let Some(context) = gpu_context() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let compiler = JitCompiler::new();
        let gemm = F32Gemm::load(&context, &compiler).unwrap();
        let stream = Stream::new(&context).unwrap();

        // Zero batches is outside the contract.
        let one = DeviceBuffer::<f32>::new(&context, 1).unwrap();
        let error = gemm
            .gemm_batched(&stream, &one, &one, &one, 0, 1, 1, 1)
            .unwrap_err();
        assert!(error.to_string().contains("at least one batch"), "{error}");

        // The gridDim.z limit is enforced before launch; 65_536 batches of
        // 1x1x1 stay allocation-affordable.
        let big = DeviceBuffer::<f32>::new(&context, 65_536).unwrap();
        let out_big = DeviceBuffer::<f32>::new(&context, 65_536).unwrap();
        let error = gemm
            .gemm_batched(&stream, &big, &big, &out_big, 65_536, 1, 1, 1)
            .unwrap_err();
        assert!(error.to_string().contains("gridDim.z limit"), "{error}");

        // Short packed operand rejected with the batch count in the message.
        let short_a = DeviceBuffer::<f32>::new(&context, 2 * 2 * 3 - 1).unwrap(); // needs 12
        let b = DeviceBuffer::<f32>::new(&context, 2 * 3 * 4).unwrap(); // needs 24
        let out = DeviceBuffer::<f32>::new(&context, 2 * 2 * 4).unwrap(); // needs 16
        let error = gemm
            .gemm_batched(&stream, &short_a, &b, &out, 2, 2, 4, 3)
            .unwrap_err();
        assert!(
            error.to_string().contains("2 batches of shape (2, 3)"),
            "{error}"
        );
    }
}
