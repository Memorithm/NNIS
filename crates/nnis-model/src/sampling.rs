use nnis_rt::{NnisError, Result};
use std::cmp::Ordering;

/// Version of NNIS's host-visible sampling semantics.
pub const NNIS_SAMPLING_POLICY_VERSION: u32 = 1;

/// Reproducible host-side sampling policy for decoder logits.
///
/// The filters are applied in this order:
/// 1. temperature scaling,
/// 2. top-k truncation by descending logit with lower token IDs winning ties,
/// 3. top-p (nucleus) truncation over the remaining normalized probabilities,
/// 4. one SplitMix64 draw from the retained distribution.
///
/// Sampling is intentionally host-visible in NNML1. Moving candidate selection
/// and RNG fully onto the device belongs to NNML2 and must not be inferred from
/// this API.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: u64,
}

impl SamplingConfig {
    /// Full-softmax seeded sampling at temperature 1.0.
    pub const fn seeded(seed: u64) -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            seed,
        }
    }

    pub const fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub const fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        if vocab_size == 0 {
            return Err(NnisError::invalid_input(
                "sampling requires a non-zero vocabulary",
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(NnisError::invalid_input(format!(
                "sampling temperature must be finite and positive; got {}",
                self.temperature
            )));
        }
        if let Some(top_k) = self.top_k {
            if top_k == 0 || top_k > vocab_size {
                return Err(NnisError::invalid_input(format!(
                    "sampling top_k must be in 1..={vocab_size}; got {top_k}"
                )));
            }
        }
        if let Some(top_p) = self.top_p {
            if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
                return Err(NnisError::invalid_input(format!(
                    "sampling top_p must be finite and in (0, 1]; got {top_p}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Candidate {
    token: u32,
    logit: f32,
    weight: f64,
}

/// Stateful deterministic sampler used by [`crate::InferenceSession`].
#[derive(Debug)]
pub(crate) struct HostLogitSampler {
    config: SamplingConfig,
    rng: SplitMix64,
    vocab_size: usize,
}

impl HostLogitSampler {
    pub(crate) fn new(config: SamplingConfig, vocab_size: usize) -> Result<Self> {
        config.validate(vocab_size)?;
        if vocab_size > u32::MAX as usize {
            return Err(NnisError::unsupported(
                "host sampler currently requires vocabulary size <= u32::MAX",
            ));
        }
        Ok(Self {
            config,
            rng: SplitMix64::new(config.seed),
            vocab_size,
        })
    }

    pub(crate) fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.len() != self.vocab_size {
            return Err(NnisError::invalid_input(format!(
                "sampling expected {} logits; got {}",
                self.vocab_size,
                logits.len()
            )));
        }
        if let Some((index, value)) = logits
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(NnisError::invalid_input(format!(
                "sampling logit {index} is non-finite: {value}"
            )));
        }

        let mut candidates = logits
            .iter()
            .copied()
            .enumerate()
            .map(|(token, logit)| Candidate {
                token: token as u32,
                logit,
                weight: 0.0,
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| {
            right
                .logit
                .partial_cmp(&left.logit)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.token.cmp(&right.token))
        });

        if let Some(top_k) = self.config.top_k {
            candidates.truncate(top_k);
        }

        let inverse_temperature = 1.0_f64 / self.config.temperature as f64;
        let max_scaled = candidates[0].logit as f64 * inverse_temperature;
        let mut total_weight = 0.0_f64;
        for candidate in &mut candidates {
            let scaled = candidate.logit as f64 * inverse_temperature;
            let weight = (scaled - max_scaled).exp();
            if !weight.is_finite() || weight < 0.0 {
                return Err(NnisError::invalid_input(
                    "sampling softmax produced a non-finite probability weight",
                ));
            }
            candidate.weight = weight;
            total_weight += weight;
        }
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(NnisError::invalid_input(
                "sampling softmax produced no positive finite probability mass",
            ));
        }

        if let Some(top_p) = self.config.top_p {
            let threshold = top_p as f64 * total_weight;
            let mut cumulative = 0.0_f64;
            let mut retained = candidates.len();
            for (index, candidate) in candidates.iter().enumerate() {
                cumulative += candidate.weight;
                if cumulative >= threshold {
                    retained = index + 1;
                    break;
                }
            }
            candidates.truncate(retained.max(1));
            total_weight = candidates.iter().map(|candidate| candidate.weight).sum();
        }

        let draw = self.rng.next_unit_f64() * total_weight;
        let mut cumulative = 0.0_f64;
        for candidate in &candidates {
            cumulative += candidate.weight;
            if draw < cumulative {
                return Ok(candidate.token);
            }
        }
        candidates
            .last()
            .map(|candidate| candidate.token)
            .ok_or_else(|| NnisError::invalid_input("sampling retained no candidates"))
    }
}

/// Small fixed algorithm so seeded sampling does not depend on a third-party RNG
/// implementation changing underneath NNIS.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(Self::GAMMA);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_unit_f64(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        mantissa as f64 * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_sampling_policies_fail_closed() {
        assert!(SamplingConfig::seeded(1).validate(4).is_ok());
        assert!(SamplingConfig::seeded(1).validate(0).is_err());
        assert!(SamplingConfig::seeded(1)
            .with_temperature(0.0)
            .validate(4)
            .is_err());
        assert!(SamplingConfig::seeded(1)
            .with_temperature(f32::NAN)
            .validate(4)
            .is_err());
        assert!(SamplingConfig::seeded(1).with_top_k(0).validate(4).is_err());
        assert!(SamplingConfig::seeded(1).with_top_k(5).validate(4).is_err());
        assert!(SamplingConfig::seeded(1).with_top_p(0.0).validate(4).is_err());
        assert!(SamplingConfig::seeded(1).with_top_p(1.1).validate(4).is_err());
        assert!(SamplingConfig::seeded(1)
            .with_top_p(f32::INFINITY)
            .validate(4)
            .is_err());
    }

    #[test]
    fn top_k_one_matches_deterministic_argmax_tie_break() {
        let mut sampler =
            HostLogitSampler::new(SamplingConfig::seeded(7).with_top_k(1), 4).unwrap();
        assert_eq!(sampler.sample(&[1.0, 4.0, 4.0, -2.0]).unwrap(), 1);
    }

    #[test]
    fn splitmix_seed_freezes_equal_logit_sequence() {
        let mut sampler = HostLogitSampler::new(SamplingConfig::seeded(42), 4).unwrap();
        let logits = [0.0_f32; 4];
        let sampled = (0..10)
            .map(|_| sampler.sample(&logits).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(sampled, vec![2, 0, 1, 1, 0, 3, 0, 3, 1, 2]);
    }

    #[test]
    fn nucleus_filter_always_keeps_the_crossing_candidate() {
        let config = SamplingConfig::seeded(9)
            .with_temperature(1.0)
            .with_top_p(0.1);
        let mut sampler = HostLogitSampler::new(config, 3).unwrap();
        for _ in 0..32 {
            assert_eq!(sampler.sample(&[3.0, 2.0, 1.0]).unwrap(), 0);
        }
    }

    #[test]
    fn top_k_limits_sampling_support() {
        let mut sampler =
            HostLogitSampler::new(SamplingConfig::seeded(123).with_top_k(2), 4).unwrap();
        for _ in 0..64 {
            let token = sampler.sample(&[4.0, 3.0, 2.0, 1.0]).unwrap();
            assert!(token <= 1);
        }
    }

    #[test]
    fn malformed_logit_vectors_are_rejected() {
        let mut sampler = HostLogitSampler::new(SamplingConfig::seeded(1), 2).unwrap();
        assert!(sampler.sample(&[1.0]).is_err());
        assert!(sampler.sample(&[1.0, f32::NAN]).is_err());
        assert!(sampler.sample(&[1.0, f32::INFINITY]).is_err());
    }
}
