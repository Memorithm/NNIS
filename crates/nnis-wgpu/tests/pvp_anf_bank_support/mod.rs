//! Frozen corpus shared by CPU and WGPU tests, not production model semantics.
//! See docs/PVP_ANF_BANK_QUALIFICATION_PROTOCOL.md.

pub const GEOMETRIES: [(usize, usize); 8] = [
    (1, 1),
    (2, 31),
    (4, 65),
    (8, 129),
    (64, 257),
    (256, 513),
    (2048, 257),
    (256, 2048),
];
pub const BANKS: [&str; 3] = ["boundary", "sparse4", "dense32"];

pub struct AnfBank {
    k: usize,
    terms: Vec<Vec<usize>>,
}

impl AnfBank {
    pub fn frozen(k: usize, gates: usize, kind: &str) -> Self {
        let terms = (0..gates)
            .map(|g| match kind {
                "boundary" => match g % 6 {
                    0 => vec![],
                    1 => vec![0],
                    2 => vec![usize::from(k > 1)],
                    3 => vec![k - 1],
                    4 => vec![0, usize::from(k > 1)],
                    _ => (0..k.ilog2()).map(|bit| 1_usize << bit).collect(),
                },
                "sparse4" | "dense32" => {
                    let count = k.min(if kind == "sparse4" { 4 } else { 32 });
                    let masks: Vec<_> = (0..count).map(|t| (257 * g + 73 * t) % k).collect();
                    let mut seen = vec![false; k];
                    for &mask in &masks {
                        assert!(!seen[mask], "duplicate frozen ANF term");
                        seen[mask] = true;
                    }
                    masks
                }
                _ => panic!("unknown frozen bank"),
            })
            .collect();
        Self { k, terms }
    }

    pub fn coefficient_words(&self) -> Vec<u64> {
        let row_words = self.terms.len().div_ceil(64);
        let mut words = vec![0_u64; self.k * row_words];
        for (gate, terms) in self.terms.iter().enumerate() {
            for &mask in terms {
                words[mask * row_words + gate / 64] ^= 1_u64 << (gate % 64);
            }
        }
        words
    }

    pub fn truth_words(&self) -> Vec<u64> {
        // Independent monomial evaluation and packing: no runtime transform
        // or layout conversion constructs the expected output.
        let row_words = self.terms.len().div_ceil(64);
        let mut words = vec![0_u64; self.k * row_words];
        for address in 0..self.k {
            for (gate, terms) in self.terms.iter().enumerate() {
                let truth = terms
                    .iter()
                    .fold(false, |value, &mask| value ^ ((address & mask) == mask));
                if truth {
                    words[address * row_words + gate / 64] |= 1_u64 << (gate % 64);
                }
            }
        }
        words
    }
}
