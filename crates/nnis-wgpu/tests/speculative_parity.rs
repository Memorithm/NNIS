//! DSV41-4 greedy speculative verification on WGPU compared with the CPU reference.
//!
//! Without an adapter each test writes an explicit SKIP to stderr and passes;
//! that pass is not evidence. Software adapters exercise the API path only and
//! are not hardware evidence. No latency, cost or throughput is measured.

use std::io::Write;
use std::sync::OnceLock;

use nnis_core::speculative_verify::{
    ConfidenceScheduleV1, SpeculativeAcceptanceStatsV1, SpeculativeDraftV1,
};
use nnis_core::{BufferDesc, BufferUsages, MemoryClass, PortableDevice, PortableQueue};
use nnis_cpu::speculative::{CpuGreedySpeculativeVerifierV1, CpuSpeculativeError};
use nnis_wgpu::speculative::{
    WgpuGreedySpeculativeVerifierV1, WgpuSpeculativeError, WGSL_GREEDY_ARGMAX_CONTRACT_VERSION,
};
use nnis_wgpu::{WgpuBuffer, WgpuDevice};

fn device() -> Option<&'static WgpuDevice> {
    static DEVICE: OnceLock<Option<WgpuDevice>> = OnceLock::new();
    DEVICE
        .get_or_init(|| WgpuDevice::discover().unwrap())
        .as_ref()
}

fn adapter_or_skip(test: &str) -> Option<&'static WgpuDevice> {
    let found = device();
    let _ = match found {
        Some(device) => writeln!(
            std::io::stderr(),
            "nnis-wgpu {test}: adapter {:?} class={:?}{}",
            device.adapter().name,
            device.adapter().class,
            if device.adapter().is_hardware() {
                ""
            } else {
                " (not hardware evidence)"
            }
        ),
        None => writeln!(
            std::io::stderr(),
            "SKIP nnis-wgpu {test}: no WGPU adapter available; \
             no WGPU execution was performed and this pass is not evidence"
        ),
    };
    found
}

fn cpu_error(error: CpuSpeculativeError) -> WgpuSpeculativeError {
    match error {
        CpuSpeculativeError::Contract(error) => WgpuSpeculativeError::Contract(error),
        CpuSpeculativeError::ZeroVocabulary => WgpuSpeculativeError::ZeroVocabulary,
        CpuSpeculativeError::TokenOutOfVocabulary { index, token } => {
            WgpuSpeculativeError::TokenOutOfVocabulary { index, token }
        }
        CpuSpeculativeError::LogitsLengthMismatch { expected, actual } => {
            WgpuSpeculativeError::LogitsLengthMismatch { expected, actual }
        }
        CpuSpeculativeError::NonFiniteLogit { row, column } => {
            WgpuSpeculativeError::NonFiniteLogit { row, column }
        }
        CpuSpeculativeError::HostIndexOverflow => WgpuSpeculativeError::HostIndexOverflow,
    }
}

/// Test-local statement of the CPU rule: finite only, strict `>`, lowest index.
fn reference_argmax(row: &[f32]) -> u32 {
    let mut best = 0;
    for (index, &value) in row.iter().enumerate() {
        if value > row[best] {
            best = index;
        }
    }
    best as u32
}

fn schedule(min_confidence: f32, max_verify: u32) -> ConfidenceScheduleV1 {
    ConfidenceScheduleV1::new("caller.threshold", 1, min_confidence, max_verify).unwrap()
}

fn upload(device: &WgpuDevice, bytes: &[u8], usages: BufferUsages) -> WgpuBuffer {
    let mut buffer = device
        .create_buffer(
            BufferDesc::new(bytes.len() as u64, usages, MemoryClass::DeviceLocal).unwrap(),
        )
        .unwrap();
    device
        .create_queue()
        .unwrap()
        .write_buffer(&mut buffer, 0, bytes)
        .unwrap();
    buffer
}

fn encode(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn all_usages() -> BufferUsages {
    BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: u32) -> u32 {
        (self.next() % u64::from(bound)) as u32
    }

    /// Finite logit drawn from a small palette so ties, signed zeros and
    /// subnormals are frequent, or from arbitrary finite bit patterns.
    fn logit(&mut self) -> f32 {
        const PALETTE: [f32; 10] = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            2.5,
            f32::MAX,
            -f32::MAX,
            f32::MIN_POSITIVE,
            1.0e-45,
            -1.0e-45,
        ];
        if self.below(3) == 0 {
            loop {
                let value = f32::from_bits(self.next() as u32);
                if value.is_finite() {
                    return value;
                }
            }
        }
        PALETTE[self.below(PALETTE.len() as u32) as usize]
    }
}

#[test]
fn randomized_drafts_verify_like_cpu() {
    let Some(device) = adapter_or_skip("randomized_drafts_verify_like_cpu") else {
        return;
    };
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let mut stats_cpu = SpeculativeAcceptanceStatsV1::default();
    let mut stats_wgpu = SpeculativeAcceptanceStatsV1::default();
    for case in 0..160 {
        let vocab = [1, 2, 3, 7, 64, 1000][case % 6];
        let cpu = CpuGreedySpeculativeVerifierV1::new(vocab).unwrap();
        let wgpu = WgpuGreedySpeculativeVerifierV1::new(vocab).unwrap();
        let len = 1 + rng.below(8) as usize;
        let tokens: Vec<u32> = (0..len).map(|_| rng.below(vocab)).collect();
        let confidences: Vec<f32> = (0..len).map(|_| rng.below(11) as f32 / 10.0).collect();
        let draft = SpeculativeDraftV1::new(tokens, confidences).unwrap();
        let schedule = schedule(rng.below(11) as f32 / 10.0, 1 + rng.below(8));
        let rows = schedule.scheduled_len(&draft) as usize + 1;
        let mut logits: Vec<f32> = (0..rows * vocab as usize).map(|_| rng.logit()).collect();
        // Plant draft agreement on some rows so accepts and rejects both occur.
        for (row, &token) in draft.tokens().iter().enumerate().take(rows - 1) {
            if rng.below(3) != 0 {
                logits[row * vocab as usize + token as usize] = f32::MAX;
            }
        }
        let expected = cpu.verify(&draft, &schedule, &logits).unwrap();
        let buffer = upload(device, &encode(&logits), all_usages());
        let outcome = wgpu.verify(device, &draft, &schedule, &buffer).unwrap();
        assert_eq!(outcome.verification, expected, "case {case}");
        let tokens: Vec<u32> = logits
            .chunks_exact(vocab as usize)
            .map(reference_argmax)
            .collect();
        assert_eq!(outcome.target_tokens, tokens, "case {case}");
        assert_eq!(outcome.schema_version, WGSL_GREEDY_ARGMAX_CONTRACT_VERSION);
        assert_ne!(outcome.artifact_fingerprint, [0; 32]);
        assert_eq!(
            wgpu.verify_host(device, &draft, &schedule, &logits)
                .unwrap()
                .verification,
            expected
        );
        cpu.verify_and_record(&mut stats_cpu, &draft, &schedule, &logits)
            .unwrap();
        wgpu.verify_and_record(device, &mut stats_wgpu, &draft, &schedule, &buffer)
            .unwrap();
        assert_eq!(stats_wgpu, stats_cpu);
    }
}

/// (vocabulary, draft tokens, confidences, threshold, logits).
type ReferenceCase = (u32, Vec<u32>, Vec<f32>, f32, Vec<f32>);

#[test]
fn reference_cases_and_ties_match_cpu() {
    let Some(device) = adapter_or_skip("reference_cases_and_ties_match_cpu") else {
        return;
    };
    let one_hot = |vocab: u32, tokens: &[u32]| -> Vec<f32> {
        tokens
            .iter()
            .flat_map(|&token| (0..vocab).map(move |c| if c == token { 2.0 } else { -1.0 }))
            .collect()
    };
    let cases: Vec<ReferenceCase> = vec![
        (
            4,
            vec![1, 2, 3],
            vec![0.9; 3],
            0.5,
            one_hot(4, &[1, 0, 3, 2]),
        ),
        (4, vec![1, 2], vec![1.0; 2], 0.5, one_hot(4, &[1, 2, 3])),
        (
            4,
            vec![2],
            vec![1.0],
            0.5,
            vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            4,
            vec![1, 2, 3],
            vec![0.9, 0.3, 0.9],
            0.5,
            one_hot(4, &[1, 3]),
        ),
        (2, vec![1], vec![1.0], 0.5, vec![-0.0, 0.0, 0.0, -0.0]),
        (
            3,
            vec![0],
            vec![1.0],
            0.5,
            vec![1.0e-45, 2.0e-45, 2.0e-45, -1.0e-45, -0.0, 0.0],
        ),
        (
            3,
            vec![2],
            vec![1.0],
            0.5,
            vec![-f32::MAX, -3.0, -3.0, 5.0, 5.0, 5.0],
        ),
    ];
    for (vocab, tokens, confidences, threshold, logits) in cases {
        let draft = SpeculativeDraftV1::new(tokens, confidences).unwrap();
        let schedule = schedule(threshold, 8);
        let expected = CpuGreedySpeculativeVerifierV1::new(vocab)
            .unwrap()
            .verify(&draft, &schedule, &logits)
            .unwrap();
        let outcome = WgpuGreedySpeculativeVerifierV1::new(vocab)
            .unwrap()
            .verify_host(device, &draft, &schedule, &logits)
            .unwrap();
        assert_eq!(outcome.verification, expected);
    }
}

#[test]
fn large_vocabulary_rows_match_cpu() {
    let Some(device) = adapter_or_skip("large_vocabulary_rows_match_cpu") else {
        return;
    };
    let vocab = 50_000u32;
    let mut rng = Rng(0xfeed_face_cafe_beef);
    let draft = SpeculativeDraftV1::new(vec![7, 49_999, 123, 0], vec![1.0; 4]).unwrap();
    let schedule = schedule(0.5, 8);
    let mut logits: Vec<f32> = (0..5 * vocab as usize)
        .map(|_| (rng.below(2_000_001) as f32 - 1.0e6) * 1.0e-3)
        .collect();
    logits[7] = 1.0e4;
    logits[vocab as usize + 49_999] = 1.0e4;
    logits[2 * vocab as usize + 124] = 1.0e4;
    let expected = CpuGreedySpeculativeVerifierV1::new(vocab)
        .unwrap()
        .verify(&draft, &schedule, &logits)
        .unwrap();
    let outcome = WgpuGreedySpeculativeVerifierV1::new(vocab)
        .unwrap()
        .verify_host(device, &draft, &schedule, &logits)
        .unwrap();
    assert_eq!(outcome.verification, expected);
    assert_eq!(expected.accepted, 2);
}

#[test]
fn errors_match_cpu_and_leave_stats_unchanged() {
    let Some(device) = adapter_or_skip("errors_match_cpu_and_leave_stats_unchanged") else {
        return;
    };
    assert_eq!(
        WgpuGreedySpeculativeVerifierV1::new(0).unwrap_err(),
        cpu_error(CpuGreedySpeculativeVerifierV1::new(0).unwrap_err())
    );
    let cpu = CpuGreedySpeculativeVerifierV1::new(4).unwrap();
    let wgpu = WgpuGreedySpeculativeVerifierV1::new(4).unwrap();
    let draft = SpeculativeDraftV1::new(vec![1, 2], vec![1.0, 1.0]).unwrap();
    let schedule = schedule(0.5, 8);
    let finite = vec![0.5f32; 12];
    let with = |edits: &[(usize, f32)]| {
        let mut logits = finite.clone();
        for &(index, value) in edits {
            logits[index] = value;
        }
        logits
    };
    let cases: Vec<(SpeculativeDraftV1, Vec<f32>)> = vec![
        (
            SpeculativeDraftV1::new(vec![1, 4], vec![1.0, 1.0]).unwrap(),
            finite.clone(),
        ),
        (draft.clone(), finite[..8].to_vec()),
        (draft.clone(), [finite.clone(), vec![0.0]].concat()),
        (draft.clone(), with(&[(6, f32::NAN), (7, f32::INFINITY)])),
        (
            draft.clone(),
            with(&[(11, f32::NEG_INFINITY), (9, f32::NAN)]),
        ),
        (draft.clone(), with(&[(0, f32::INFINITY)])),
        // A non-finite logit after the first mismatch still fails closed.
        (draft.clone(), with(&[(3, 9.0), (10, f32::NAN)])),
    ];
    for (draft, logits) in cases {
        let expected = cpu_error(cpu.verify(&draft, &schedule, &logits).unwrap_err());
        assert_eq!(
            wgpu.verify_host(device, &draft, &schedule, &logits)
                .unwrap_err(),
            expected
        );
        if logits.len() == 12
            && !matches!(expected, WgpuSpeculativeError::TokenOutOfVocabulary { .. })
        {
            let buffer = upload(device, &encode(&logits), all_usages());
            let mut stats = SpeculativeAcceptanceStatsV1::default();
            assert_eq!(
                wgpu.verify_and_record(device, &mut stats, &draft, &schedule, &buffer)
                    .unwrap_err(),
                expected
            );
            assert_eq!(stats, SpeculativeAcceptanceStatsV1::default());
        }
    }
    // Device-buffer length and usage checks.
    assert_eq!(
        wgpu.verify(
            device,
            &draft,
            &schedule,
            &upload(device, &encode(&finite[..8]), all_usages())
        )
        .unwrap_err(),
        WgpuSpeculativeError::LogitsLengthMismatch {
            expected: 12,
            actual: 8,
        }
    );
    assert_eq!(
        wgpu.verify(
            device,
            &draft,
            &schedule,
            &upload(device, &[0u8; 47], all_usages())
        )
        .unwrap_err(),
        WgpuSpeculativeError::LogitsByteLengthMismatch {
            expected_bytes: 48,
            actual_bytes: 47,
        }
    );
    assert!(matches!(
        wgpu.verify(
            device,
            &draft,
            &schedule,
            &upload(
                device,
                &encode(&finite),
                BufferUsages::COPY_SRC | BufferUsages::COPY_DST
            )
        ),
        Err(WgpuSpeculativeError::Portable(_))
    ));
}
