from pathlib import Path

p = Path("crates/nnis-model/src/da_luc_plan.rs")
s = p.read_text()
old = """#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = \"policy\", content = \"candidate\", rename_all = \"snake_case\")]
pub enum NnisKvExecutionPolicy {
    DenseReference,
    DaLucCandidate(NnisDalucCandidatePlan),
}

impl Default for NnisKvExecutionPolicy {
    fn default() -> Self {
        Self::DenseReference
    }
}
"""
new = """#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = \"policy\", content = \"candidate\", rename_all = \"snake_case\")]
pub enum NnisKvExecutionPolicy {
    #[default]
    DenseReference,
    DaLucCandidate(NnisDalucCandidatePlan),
}
"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """        value_group_size: usize,
    ) -> Self {
        let head_dim = config.head_dim();
        Self {
"""
new = """        value_group_size: usize,
    ) -> Result<Self> {
        config.validate_execution_support()?;
        let head_dim = config.head_dim();
        Ok(Self {
"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """            physical_layout: NnisDalucCudaPhysicalLayout::Fdal3CompatibleWordPackedV1,
        }
    }

    pub fn validate(
"""
new = """            physical_layout: NnisDalucCudaPhysicalLayout::Fdal3CompatibleWordPackedV1,
        })
    }

    pub fn validate(
"""
assert s.count(old) == 1
s = s.replace(old, new)
old = """            NnisDalucCodebookScope::PerKvHead,
            8,
        )
    }

    #[test]
    fn default_execution_policy_remains_dense_reference() {
"""
new = """            NnisDalucCodebookScope::PerKvHead,
            8,
        )
        .unwrap()
    }

    #[test]
    fn constructor_rejects_invalid_model_geometry_without_panicking() {
        let mut invalid = config();
        invalid.num_attention_heads = 0;
        assert!(NnisDalucCandidatePlan::fdal3_compatible_v1(
            &invalid,
            256,
            8,
            64,
            NnisDalucCodebookScope::PerKvHead,
            8,
        )
        .is_err());
    }

    #[test]
    fn default_execution_policy_remains_dense_reference() {
"""
assert s.count(old) == 1
s = s.replace(old, new)
p.write_text(s)
