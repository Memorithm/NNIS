//! CPU reference execution session for Pascal Vector Projection (PVP).
//!
//! This module consumes the backend-neutral PVP layout adapter from nnis-core
//! and executes the exact subset-zeta/Pascal XOR butterfly over the canonical
//! CPU-u64 physical projection.
//!
//! It is a runtime carrier only:
//! - no model/ANF semantics;
//! - no GPU API;
//! - no vendor SDK;
//! - no performance claim.
//!
//! SciRust remains the reusable SIMD/oracle owner. NNIS-PVP1 establishes an
//! independently testable CPU runtime session and accounting surface that can
//! later be compared with the WGPU session.

use nnis_core::pvp::{PvpCpuU64V1, PvpLayoutAdapterV1, PvpLayoutError, PvpPhysicalWordV1};

/// Stable identity of the NNIS CPU PVP execution contract.
pub const NNIS_PVP_CPU_EXECUTION_SCHEMA_V1: &str = "nnis.pvp-cpu-session.v1";
/// Stable backend identity used by the portable CPU PVP session.
pub const NNIS_PVP_CPU_BACKEND_ID: &str = "nnis-cpu-reference";

/// Exact accounting for one complete CPU PVP transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuPvpExecutionStatsV1 {
    pub stages: u32,
    pub logical_gate_xor_ops: u128,
    pub packed_u64_updates: u128,
    pub state_words: usize,
    pub state_bytes: usize,
    pub scratch_words: usize,
    pub execution_index: u64,
}

/// Stateful CPU runtime carrier for one canonical PVP bank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuPvpSessionV1 {
    layout: PvpLayoutAdapterV1,
    words: Vec<u64>,
    executions: u64,
}

impl CpuPvpSessionV1 {
    /// Start a session from already validated CPU-u64 PVP storage.
    #[must_use]
    pub fn new(initial: &PvpCpuU64V1) -> Self {
        Self {
            layout: initial.layout(),
            words: initial.words().to_vec(),
            executions: 0,
        }
    }

    /// Validate raw CPU-u64 storage and start a session.
    pub fn from_words(layout: PvpLayoutAdapterV1, words: Vec<u64>) -> Result<Self, PvpLayoutError> {
        let initial = PvpCpuU64V1::new(layout, words)?;
        Ok(Self {
            layout,
            words: initial.words().to_vec(),
            executions: 0,
        })
    }

    /// Logical layout of the resident bank.
    #[must_use]
    pub const fn layout(&self) -> PvpLayoutAdapterV1 {
        self.layout
    }

    /// Read-only resident CPU-u64 state.
    #[must_use]
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Number of full PVP transforms executed by this session.
    #[must_use]
    pub const fn execution_count(&self) -> u64 {
        self.executions
    }

    /// Return a revalidated owned snapshot of the current state.
    pub fn snapshot(&self) -> Result<PvpCpuU64V1, PvpLayoutError> {
        PvpCpuU64V1::new(self.layout, self.words.clone())
    }

    /// Execute the exact subset-zeta/Pascal transform in place.
    ///
    /// The transform is self-inverse over GF(2). No algorithmic scratch buffer
    /// is allocated. The resident state itself was allocated when the session
    /// was constructed.
    pub fn execute_subset_zeta(&mut self) -> Result<CpuPvpExecutionStatsV1, PvpLayoutError> {
        let words_per_address = self.layout.words_per_address(PvpPhysicalWordV1::CpuU64)?;
        let mut stride = 1_usize;

        while stride < self.layout.addresses() {
            let block = stride
                .checked_mul(2)
                .ok_or(PvpLayoutError::ArithmeticOverflow)?;
            let mut block_start = 0_usize;
            while block_start < self.layout.addresses() {
                for offset in 0..stride {
                    let source_address = block_start + offset;
                    let target_address = source_address + stride;
                    let source_base = source_address
                        .checked_mul(words_per_address)
                        .ok_or(PvpLayoutError::ArithmeticOverflow)?;
                    let target_base = target_address
                        .checked_mul(words_per_address)
                        .ok_or(PvpLayoutError::ArithmeticOverflow)?;
                    for word in 0..words_per_address {
                        self.words[target_base + word] ^= self.words[source_base + word];
                    }
                }
                block_start = block_start
                    .checked_add(block)
                    .ok_or(PvpLayoutError::ArithmeticOverflow)?;
            }
            stride = block;
        }

        // Revalidate the public representation contract after mutation. XOR of
        // canonical rows preserves zero tail padding; fail closed if that
        // invariant is ever violated by a future implementation change.
        let _ = PvpCpuU64V1::new(self.layout, self.words.clone())?;

        self.executions = self
            .executions
            .checked_add(1)
            .ok_or(PvpLayoutError::ArithmeticOverflow)?;

        let pairs = (self.layout.addresses() / 2) as u128 * u128::from(self.layout.stages());
        Ok(CpuPvpExecutionStatsV1 {
            stages: self.layout.stages(),
            logical_gate_xor_ops: pairs * self.layout.gates() as u128,
            packed_u64_updates: pairs * words_per_address as u128,
            state_words: self.layout.storage_words(PvpPhysicalWordV1::CpuU64)?,
            state_bytes: self.layout.storage_bytes(PvpPhysicalWordV1::CpuU64)?,
            scratch_words: 0,
            execution_index: self.executions,
        })
    }

    /// Stable runtime identity bound to this exact resident layout.
    pub fn canonical_record(&self) -> Result<String, PvpLayoutError> {
        Ok(format!(
            "{};backend={};executions={};{}",
            NNIS_PVP_CPU_EXECUTION_SCHEMA_V1,
            NNIS_PVP_CPU_BACKEND_ID,
            self.executions,
            self.layout.canonical_record(PvpPhysicalWordV1::CpuU64)?
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(layout: PvpLayoutAdapterV1) -> PvpCpuU64V1 {
        let row_words = layout.words_per_address(PvpPhysicalWordV1::CpuU64).unwrap();
        let mut words = vec![0_u64; layout.storage_words(PvpPhysicalWordV1::CpuU64).unwrap()];
        for address in 0..layout.addresses() {
            for gate in 0..layout.gates() {
                if ((address * 17 + gate * 13 + (address ^ gate)) % 11) < 5 {
                    words[address * row_words + gate / 64] |= 1_u64 << (gate % 64);
                }
            }
        }
        PvpCpuU64V1::new(layout, words).unwrap()
    }

    fn direct_subset_oracle(source: &PvpCpuU64V1) -> Vec<u64> {
        let layout = source.layout();
        let row_words = layout.words_per_address(PvpPhysicalWordV1::CpuU64).unwrap();
        let mut output = vec![0_u64; source.words().len()];

        for address in 0..layout.addresses() {
            let target_base = address * row_words;
            let mut submask = address;
            loop {
                let source_base = submask * row_words;
                for word in 0..row_words {
                    output[target_base + word] ^= source.words()[source_base + word];
                }
                if submask == 0 {
                    break;
                }
                submask = (submask - 1) & address;
            }
        }

        output
    }

    #[test]
    fn cpu_session_matches_independent_direct_subset_oracle() {
        let layout = PvpLayoutAdapterV1::new(16, 70).unwrap();
        let source = fixture(layout);
        let expected = direct_subset_oracle(&source);
        let mut session = CpuPvpSessionV1::new(&source);

        let stats = session.execute_subset_zeta().unwrap();

        assert_eq!(session.words(), expected.as_slice());
        assert_eq!(stats.stages, 4);
        assert_eq!(stats.logical_gate_xor_ops, 2_240);
        assert_eq!(stats.packed_u64_updates, 64);
        assert_eq!(stats.state_words, 32);
        assert_eq!(stats.state_bytes, 256);
        assert_eq!(stats.scratch_words, 0);
        assert_eq!(stats.execution_index, 1);
        assert_eq!(session.execution_count(), 1);
    }

    #[test]
    fn cpu_session_transform_is_self_inverse() {
        let layout = PvpLayoutAdapterV1::new(64, 129).unwrap();
        let source = fixture(layout);
        let mut session = CpuPvpSessionV1::new(&source);

        session.execute_subset_zeta().unwrap();
        session.execute_subset_zeta().unwrap();

        assert_eq!(session.snapshot().unwrap(), source);
        assert_eq!(session.execution_count(), 2);
    }

    #[test]
    fn cpu_session_preserves_canonical_padding_and_round_trip_adapter() {
        let layout = PvpLayoutAdapterV1::new(32, 65).unwrap();
        let source = fixture(layout);
        let mut session = CpuPvpSessionV1::new(&source);
        session.execute_subset_zeta().unwrap();

        let snapshot = session.snapshot().unwrap();
        let portable = snapshot.to_wgpu_u32().unwrap();
        let back = portable.to_cpu_u64().unwrap();
        assert_eq!(back, snapshot);
    }

    #[test]
    fn canonical_record_binds_backend_layout_and_execution_count() {
        let layout = PvpLayoutAdapterV1::new(8, 33).unwrap();
        let source = fixture(layout);
        let mut session = CpuPvpSessionV1::new(&source);
        session.execute_subset_zeta().unwrap();

        let record = session.canonical_record().unwrap();
        assert!(record.contains(NNIS_PVP_CPU_EXECUTION_SCHEMA_V1));
        assert!(record.contains(NNIS_PVP_CPU_BACKEND_ID));
        assert!(record.contains("executions=1"));
        assert!(record.contains("source_logical=pvp-bitplanes/v1"));
        assert!(record.contains("physical=cpu-u64"));
    }

    #[test]
    fn raw_constructor_rejects_noncanonical_padding() {
        let layout = PvpLayoutAdapterV1::new(8, 65).unwrap();
        let mut words = vec![0_u64; layout.storage_words(PvpPhysicalWordV1::CpuU64).unwrap()];
        words[1] = 1_u64 << 63;
        assert!(CpuPvpSessionV1::from_words(layout, words).is_err());
    }
}
