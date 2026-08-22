//! GPU device discovery and capability reporting.

use crate::error::{NnisError, Result};
use nnis_sys::constants as cu;
use nnis_sys::{driver, CUdevice};

/// Static capabilities of one NVIDIA device.
#[derive(Debug, Clone)]
pub struct DeviceProps {
    pub ordinal: i32,
    pub name: String,
    pub uuid: Option<nnis_sys::CUuuid>,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_multiprocessor: u32,
    pub max_registers_per_block: u32,
    pub shared_memory_per_block: u32,
    pub shared_memory_per_block_optin: u32,
    pub shared_memory_per_multiprocessor: u32,
    pub l2_cache_size: u32,
    pub clock_khz: u32,
    pub memory_clock_khz: u32,
    pub global_memory_bus_width_bits: u32,
    pub integrated: bool,
    pub ecc_enabled: bool,
    pub unified_addressing: bool,
}

impl DeviceProps {
    /// NVRTC / nvcc architecture string, e.g. `"sm_110"`.
    ///
    /// Derived from the compute capability; unknown future architectures
    /// still produce a well-formed string (`sm_<major><minor>`).
    pub fn sm_arch(&self) -> String {
        format!(
            "sm_{}{}",
            self.compute_capability.0, self.compute_capability.1
        )
    }

    /// One-line fingerprint useful in benchmark metadata.
    pub fn fingerprint(&self) -> String {
        format!(
            "{} cc{}.{}, {} SMs, {} kHz core",
            self.name,
            self.compute_capability.0,
            self.compute_capability.1,
            self.multiprocessor_count,
            self.clock_khz
        )
    }
}

impl std::fmt::Display for DeviceProps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.fingerprint())
    }
}

/// A handle to an installed CUDA device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    ordinal: i32,
}

impl Device {
    /// Number of visible CUDA-capable devices.
    pub fn count() -> Result<usize> {
        let api = driver::api()?;
        let mut n: core::ffi::c_int = 0;
        // SAFETY: plain query; pointer is valid.
        let rc = unsafe { (api.cuDeviceGetCount)(&mut n) };
        if rc != 0 {
            return Err(NnisError::driver("cuDeviceGetCount", rc));
        }
        Ok(n.max(0) as usize)
    }

    /// Open a device by ordinal, validating that it exists.
    pub fn get(ordinal: i32) -> Result<Self> {
        if ordinal < 0 {
            return Err(NnisError::invalid_input(format!(
                "device ordinal {ordinal} is negative"
            )));
        }
        let api = driver::api()?;
        let mut dev: CUdevice = -1;
        // SAFETY: out-pointer is valid.
        let rc = unsafe { (api.cuDeviceGet)(&mut dev, ordinal) };
        if rc != 0 {
            return Err(NnisError::driver("cuDeviceGet", rc).with("ordinal", ordinal));
        }
        Ok(Device { ordinal })
    }

    /// All visible devices, or an error explaining why none are available.
    pub fn enumerate() -> Result<Vec<Device>> {
        let n = Self::count()?;
        let mut v = Vec::with_capacity(n);
        for i in 0..n as i32 {
            v.push(Self::get(i)?);
        }
        if v.is_empty() {
            return Err(NnisError::unsupported(
                "no CUDA devices are visible to the driver",
            ));
        }
        Ok(v)
    }

    /// First available device. Preferred entry point for single-GPU hosts.
    pub fn first() -> Result<Self> {
        Self::get(0)
    }

    pub fn ordinal(&self) -> i32 {
        self.ordinal
    }

    fn attribute(&self, attr: i32) -> Result<i32> {
        let api = driver::api()?;
        let mut v: core::ffi::c_int = 0;
        // SAFETY: out-pointer is valid; attr values come from verified constants.
        let rc = unsafe { (api.cuDeviceGetAttribute)(&mut v, attr, self.ordinal) };
        if rc != 0 {
            return Err(NnisError::driver("cuDeviceGetAttribute", rc)
                .with("attribute", attr)
                .with("ordinal", self.ordinal));
        }
        Ok(v)
    }

    /// Query full static capabilities of this device.
    pub fn props(&self) -> Result<DeviceProps> {
        Ok(DeviceProps {
            ordinal: self.ordinal,
            name: self.name()?,
            uuid: self.uuid(),
            compute_capability: (
                self.attribute(cu::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                self.attribute(cu::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
            ),
            multiprocessor_count: self.attribute(cu::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
                as u32,
            warp_size: self.attribute(cu::CU_DEVICE_ATTRIBUTE_WARP_SIZE)? as u32,
            max_threads_per_block: self.attribute(cu::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?
                as u32,
            max_threads_per_multiprocessor: self
                .attribute(cu::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)?
                as u32,
            max_registers_per_block: self
                .attribute(cu::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)?
                as u32,
            shared_memory_per_block: self
                .attribute(cu::CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK)?
                as u32,
            shared_memory_per_block_optin: self
                .attribute(cu::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?
                as u32,
            shared_memory_per_multiprocessor: self
                .attribute(cu::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)?
                as u32,
            l2_cache_size: self.attribute(cu::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)? as u32,
            clock_khz: self.attribute(cu::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)? as u32,
            memory_clock_khz: self.attribute(cu::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)? as u32,
            global_memory_bus_width_bits: self
                .attribute(cu::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?
                as u32,
            integrated: self.attribute(cu::CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0,
            ecc_enabled: self.attribute(cu::CU_DEVICE_ATTRIBUTE_ECC_ENABLED)? != 0,
            unified_addressing: self.attribute(cu::CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING)? != 0,
        })
    }

    /// Device name string.
    pub fn name(&self) -> Result<String> {
        const LEN: usize = 256;
        let api = driver::api()?;
        let mut buf = [0 as core::ffi::c_char; LEN];
        // SAFETY: buffer and length are valid.
        let rc = unsafe { (api.cuDeviceGetName)(buf.as_mut_ptr(), LEN as i32, self.ordinal) };
        if rc != 0 {
            return Err(NnisError::driver("cuDeviceGetName", rc).with("ordinal", self.ordinal));
        }
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Device UUID (v2 layout when the driver exposes it).
    pub fn uuid(&self) -> Option<nnis_sys::CUuuid> {
        let api = driver::api().ok()?;
        let mut u = nnis_sys::CUuuid([0u8; 16]);
        // SAFETY: out-pointer is valid.
        let rc = unsafe { (api.cuDeviceGetUuid)(&mut u, self.ordinal) };
        if rc == 0 {
            Some(u)
        } else {
            None
        }
    }
}
