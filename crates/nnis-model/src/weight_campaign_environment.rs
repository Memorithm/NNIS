//! Physical environment identity for full-model weight qualification campaigns.

use nnis_rt::{Device, NnisError, Result};
use serde::{Deserialize, Serialize};

/// Version of the physical weight-campaign environment contract.
pub const NNIS_WEIGHT_CAMPAIGN_ENVIRONMENT_VERSION: u32 = 1;

/// Backend-owned environment fingerprint captured from the CUDA driver API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightCampaignEnvironmentV1 {
    pub schema_version: u32,
    pub device_ordinal: i32,
    pub device_name: String,
    pub device_uuid: String,
    pub compute_capability_major: i32,
    pub compute_capability_minor: i32,
    pub sm_arch: String,
    pub multiprocessor_count: u32,
    pub clock_khz: u32,
    pub memory_clock_khz: u32,
    pub integrated: bool,
    pub cuda_driver_major: i32,
    pub cuda_driver_minor: i32,
}

impl WeightCampaignEnvironmentV1 {
    /// Capture the physical CUDA environment from the exact device used by a run.
    pub fn capture(device: &Device) -> Result<Self> {
        let props = device.props()?;
        let uuid = props.uuid.ok_or_else(|| {
            NnisError::unsupported("weight campaign environment requires a CUDA device UUID")
        })?;
        let (cuda_driver_major, cuda_driver_minor) = props.driver_version.ok_or_else(|| {
            NnisError::unsupported("weight campaign environment requires CUDA driver version")
        })?;
        let environment = Self {
            schema_version: NNIS_WEIGHT_CAMPAIGN_ENVIRONMENT_VERSION,
            device_ordinal: props.ordinal,
            device_name: props.name,
            device_uuid: format!("GPU-{uuid:?}"),
            compute_capability_major: props.compute_capability.0,
            compute_capability_minor: props.compute_capability.1,
            sm_arch: props.sm_arch(),
            multiprocessor_count: props.multiprocessor_count,
            clock_khz: props.clock_khz,
            memory_clock_khz: props.memory_clock_khz,
            integrated: props.integrated,
            cuda_driver_major,
            cuda_driver_minor,
        };
        environment.validate()?;
        Ok(environment)
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != NNIS_WEIGHT_CAMPAIGN_ENVIRONMENT_VERSION {
            return Err(NnisError::unsupported(format!(
                "weight campaign environment schema {}; supported version is {}",
                self.schema_version, NNIS_WEIGHT_CAMPAIGN_ENVIRONMENT_VERSION
            )));
        }
        if self.device_ordinal < 0
            || self.device_name.is_empty()
            || self.device_name.trim() != self.device_name
            || !self.device_uuid.starts_with("GPU-")
            || self.device_uuid.len() <= 4
        {
            return Err(NnisError::invalid_input(
                "weight campaign device identity is invalid",
            ));
        }
        if self.compute_capability_major <= 0
            || self.compute_capability_minor < 0
            || self.multiprocessor_count == 0
            || self.clock_khz == 0
        {
            return Err(NnisError::invalid_input(
                "weight campaign CUDA device capabilities are invalid",
            ));
        }
        let expected_sm_arch = format!(
            "sm_{}{}",
            self.compute_capability_major, self.compute_capability_minor
        );
        if self.sm_arch != expected_sm_arch {
            return Err(NnisError::invalid_input(format!(
                "weight campaign sm_arch {:?} disagrees with compute capability {}.{}",
                self.sm_arch, self.compute_capability_major, self.compute_capability_minor
            )));
        }
        if self.cuda_driver_major <= 0 || self.cuda_driver_minor < 0 {
            return Err(NnisError::invalid_input(
                "weight campaign CUDA driver version is invalid",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> WeightCampaignEnvironmentV1 {
        WeightCampaignEnvironmentV1 {
            schema_version: NNIS_WEIGHT_CAMPAIGN_ENVIRONMENT_VERSION,
            device_ordinal: 0,
            device_name: "NVIDIA Test GPU".to_string(),
            device_uuid: "GPU-CUuuid([0, 1, 2, 3])".to_string(),
            compute_capability_major: 12,
            compute_capability_minor: 1,
            sm_arch: "sm_121".to_string(),
            multiprocessor_count: 16,
            clock_khz: 1_000_000,
            memory_clock_khz: 500_000,
            integrated: true,
            cuda_driver_major: 13,
            cuda_driver_minor: 0,
        }
    }

    #[test]
    fn environment_contract_is_versioned_and_fail_closed() {
        let environment = fixture();
        environment.validate().unwrap();
        let encoded = serde_json::to_string(&environment).unwrap();
        let decoded: WeightCampaignEnvironmentV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, environment);

        let mut drifted = environment.clone();
        drifted.sm_arch = "sm_120".to_string();
        assert!(drifted.validate().is_err());

        let mut drifted = environment;
        drifted.cuda_driver_major = 0;
        assert!(drifted.validate().is_err());
    }

    #[test]
    fn capture_uses_real_device_when_available() {
        let Ok(device) = Device::first() else {
            eprintln!("skipped: no CUDA device");
            return;
        };
        let environment = WeightCampaignEnvironmentV1::capture(&device).unwrap();
        environment.validate().unwrap();
        assert_eq!(environment.device_ordinal, device.ordinal());
    }
}
