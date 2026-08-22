//! Enum / flag constants transcribed from the installed CUDA headers.
//!
//! Source of truth: `/usr/local/cuda/include/cuda.h` (CUDA 13.0). Do not add
//! constants here without verifying them against a real header.

/// `CUstream_flags`
pub const CU_STREAM_DEFAULT: u32 = 0x0;
pub const CU_STREAM_NON_BLOCKING: u32 = 0x1;

/// `CUevent_flags`
pub const CU_EVENT_DEFAULT: u32 = 0x0;
pub const CU_EVENT_BLOCKING_SYNC: u32 = 0x1;
/// Events created with this flag cannot be used for timing.
pub const CU_EVENT_DISABLE_TIMING: u32 = 0x2;

/// `CUdevice_attribute` (subset used by NNIS)
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 1;
pub const CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK: i32 = 8;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: i32 = 10;
pub const CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: i32 = 12;
pub const CU_DEVICE_ATTRIBUTE_CLOCK_RATE: i32 = 13;
pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
pub const CU_DEVICE_ATTRIBUTE_INTEGRATED: i32 = 18;
pub const CU_DEVICE_ATTRIBUTE_ECC_ENABLED: i32 = 32;
pub const CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: i32 = 36;
pub const CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;
pub const CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: i32 = 38;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;
pub const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: i32 = 41;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: i32 = 81;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;

/// `CUfunction_attribute` (subset)
pub const CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK: i32 = 0;
pub const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: i32 = 1;
pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: i32 = 3;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: i32 = 4;
pub const CU_FUNC_ATTRIBUTE_BINARY_VERSION: i32 = 6;
pub const CU_FUNC_ATTRIBUTE_PTX_VERSION: i32 = 5;

/// `CUjit_option` (subset used to capture JIT logs)
pub const CU_JIT_INFO_LOG_BUFFER: i32 = 3;
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: i32 = 4;
pub const CU_JIT_ERROR_LOG_BUFFER: i32 = 5;
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: i32 = 6;
pub const CU_JIT_TARGET_FROM_CUCONTEXT: i32 = 8;
pub const CU_JIT_LOG_VERBOSE: i32 = 12;

/// Well-known driver error codes referenced by name in tests/diagnostics.
pub mod error_codes {
    pub const CUDA_SUCCESS: i32 = 0;
    pub const CUDA_ERROR_INVALID_VALUE: i32 = 1;
    pub const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
    pub const CUDA_ERROR_NOT_INITIALIZED: i32 = 3;
    pub const CUDA_ERROR_NO_DEVICE: i32 = 100;
    pub const CUDA_ERROR_INVALID_CONTEXT: i32 = 201;
    pub const CUDA_ERROR_NOT_READY: i32 = 600;
    pub const CUDA_ERROR_INVALID_SOURCE: i32 = 205; // module load
    pub const CUDA_ERROR_FILE_NOT_FOUND: i32 = 301;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spot_check_against_header_values() {
        // These were verified against cuda.h from CUDA 13.0; a regression here
        // means someone edited constants without re-verifying.
        assert_eq!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 75);
        assert_eq!(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 76);
        assert_eq!(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 16);
        assert_eq!(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 38);
        assert_eq!(CU_FUNC_ATTRIBUTE_NUM_REGS, 4);
        assert_eq!(CU_JIT_ERROR_LOG_BUFFER, 5);
        assert_eq!(error_codes::CUDA_ERROR_OUT_OF_MEMORY, 2);
    }
}
