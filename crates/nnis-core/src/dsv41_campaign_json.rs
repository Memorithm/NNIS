//! Versioned JSON encoding of DSV41-5 preregistration and evidence records.
//!
//! Records carry a `kind` tag and `schema_version`. Parsing fails closed on
//! syntax errors, a wrong kind or version, unknown or missing fields, wrong
//! types, non-integer or out-of-range integers, and non-finite numbers.
//! Optional evidence metrics must be present as keys and may be `null`; a
//! `null` metric is then rejected by
//! [`validate_dsv41_evidence`](crate::dsv41_campaign::validate_dsv41_evidence).
//!
//! Parsing an evidence record does not validate it. Encoding records nothing
//! new: it serializes exactly the values supplied by the caller.

use core::fmt;

use crate::dsv41_campaign::{
    CampaignArmV1, CampaignBackendV1, CampaignCellV1, CampaignHardwareV1, CampaignMemoryMetricV1,
    CampaignModelV1, CampaignPromptSuiteV1, CampaignProvenanceV1, CampaignQualityMetricV1,
    CampaignRecordKindV1, DecodeLatencyDistributionV1, Dsv41CampaignError, Dsv41CampaignEvidenceV1,
    Dsv41CampaignPreregistrationFieldsV1, Dsv41CampaignPreregistrationV1, Dsv41PrimitiveV1,
    QualityDirectionV1,
};
use crate::json::{self, Json};

/// `kind` tag of a preregistration record.
pub const DSV41_PREREGISTRATION_KIND: &str = "nnis.dsv41.campaign-preregistration";

/// `kind` tag of an evidence record.
pub const DSV41_EVIDENCE_KIND: &str = "nnis.dsv41.campaign-evidence";

/// JSON schema version written and accepted.
pub const DSV41_CAMPAIGN_JSON_SCHEMA_VERSION: u32 = 1;

/// Encode a validated preregistration as compact JSON.
pub fn preregistration_to_json(
    record: &Dsv41CampaignPreregistrationV1,
) -> Result<String, Dsv41JsonError> {
    let model = record.model();
    let suite = record.prompt_suite();
    let quality = record.quality_metric();
    let value = Json::Object(vec![
        field("kind", text(DSV41_PREREGISTRATION_KIND)),
        field(
            "schema_version",
            json::unsigned(u64::from(DSV41_CAMPAIGN_JSON_SCHEMA_VERSION)),
        ),
        field("campaign_id", text(record.campaign_id())),
        field(
            "model",
            Json::Object(vec![
                field("model_id", text(&model.model_id)),
                field("revision", text(&model.revision)),
                field("weights_sha256", text(&model.weights_sha256)),
            ]),
        ),
        field(
            "prompt_suite",
            Json::Object(vec![
                field("suite_id", text(&suite.suite_id)),
                field("suite_sha256", text(&suite.suite_sha256)),
                field(
                    "prompt_count",
                    json::unsigned(u64::from(suite.prompt_count)),
                ),
            ]),
        ),
        field("backend", text(backend_name(record.backend()))),
        field(
            "arms",
            Json::Array(
                record
                    .arms()
                    .iter()
                    .map(|arm| {
                        Json::Object(vec![
                            field("arm_id", text(&arm.arm_id)),
                            field(
                                "primitives",
                                Json::Array(
                                    arm.primitives
                                        .iter()
                                        .map(|p| text(primitive_name(*p)))
                                        .collect(),
                                ),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        field("context_lengths", u32_array(record.context_lengths())),
        field("decode_lengths", u32_array(record.decode_lengths())),
        field(
            "repetitions",
            json::unsigned(u64::from(record.repetitions())),
        ),
        field(
            "memory_metric",
            text(memory_metric_name(record.memory_metric())),
        ),
        field(
            "quality_metric",
            Json::Object(vec![
                field("metric_id", text(&quality.metric_id)),
                field("direction", text(direction_name(quality.direction))),
                field(
                    "max_regression",
                    finite("max_regression", quality.max_regression)?,
                ),
            ]),
        ),
    ]);
    Ok(render(&value))
}

/// Parse and validate a preregistration record.
pub fn preregistration_from_json(
    input: &str,
) -> Result<Dsv41CampaignPreregistrationV1, Dsv41JsonError> {
    let mut root = Object::root(input, DSV41_PREREGISTRATION_KIND)?;
    let campaign_id = root.string("campaign_id")?;
    let mut model = root.object("model")?;
    let model_fields = CampaignModelV1 {
        model_id: model.string("model_id")?,
        revision: model.string("revision")?,
        weights_sha256: model.string("weights_sha256")?,
    };
    model.finish()?;
    let mut suite = root.object("prompt_suite")?;
    let prompt_suite = CampaignPromptSuiteV1 {
        suite_id: suite.string("suite_id")?,
        suite_sha256: suite.string("suite_sha256")?,
        prompt_count: suite.u32("prompt_count")?,
    };
    suite.finish()?;
    let backend = parse_backend("backend", &root.string("backend")?)?;
    let mut arms = Vec::new();
    for item in root.array("arms")? {
        let mut arm = Object::new("arms[]", item)?;
        let arm_id = arm.string("arm_id")?;
        let mut primitives = Vec::new();
        for primitive in arm.array("primitives")? {
            primitives.push(parse_primitive(&as_string("primitives[]", primitive)?)?);
        }
        arm.finish()?;
        arms.push(CampaignArmV1 { arm_id, primitives });
    }
    let context_lengths = root.u32_array("context_lengths")?;
    let decode_lengths = root.u32_array("decode_lengths")?;
    let repetitions = root.u32("repetitions")?;
    let memory_metric = parse_memory_metric("memory_metric", &root.string("memory_metric")?)?;
    let mut quality = root.object("quality_metric")?;
    let quality_metric = CampaignQualityMetricV1 {
        metric_id: quality.string("metric_id")?,
        direction: parse_direction(&quality.string("direction")?)?,
        max_regression: quality.f64("max_regression")?,
    };
    quality.finish()?;
    root.finish()?;
    Ok(Dsv41CampaignPreregistrationV1::new(
        Dsv41CampaignPreregistrationFieldsV1 {
            campaign_id,
            model: model_fields,
            prompt_suite,
            backend,
            arms,
            context_lengths,
            decode_lengths,
            repetitions,
            memory_metric,
            quality_metric,
        },
    )?)
}

/// Encode an evidence record as compact JSON (without validating it).
pub fn evidence_to_json(record: &Dsv41CampaignEvidenceV1) -> Result<String, Dsv41JsonError> {
    let mut cells = Vec::with_capacity(record.cells.len());
    for cell in &record.cells {
        let latency = match cell.decode_latency {
            None => Json::Null,
            Some(latency) => Json::Object(vec![
                field("samples", json::unsigned(latency.samples)),
                field("p50_ms", finite("p50_ms", latency.p50_ms)?),
                field("p90_ms", finite("p90_ms", latency.p90_ms)?),
                field("p99_ms", finite("p99_ms", latency.p99_ms)?),
                field("max_ms", finite("max_ms", latency.max_ms)?),
            ]),
        };
        cells.push(Json::Object(vec![
            field("arm_id", text(&cell.arm_id)),
            field(
                "context_length",
                json::unsigned(u64::from(cell.context_length)),
            ),
            field(
                "decode_length",
                json::unsigned(u64::from(cell.decode_length)),
            ),
            field("repetitions", json::unsigned(u64::from(cell.repetitions))),
            field("quality", optional_finite("quality", cell.quality)?),
            field(
                "memory_metric",
                cell.memory_metric
                    .map_or(Json::Null, |metric| text(memory_metric_name(metric))),
            ),
            field(
                "memory_bytes",
                cell.memory_bytes.map_or(Json::Null, json::unsigned),
            ),
            field("decode_latency", latency),
            field(
                "tokens_per_second",
                optional_finite("tokens_per_second", cell.tokens_per_second)?,
            ),
        ]));
    }
    let value = Json::Object(vec![
        field("kind", text(DSV41_EVIDENCE_KIND)),
        field(
            "schema_version",
            json::unsigned(u64::from(DSV41_CAMPAIGN_JSON_SCHEMA_VERSION)),
        ),
        field(
            "record_kind",
            text(match record.kind {
                CampaignRecordKindV1::PhysicalExecution => "physical_execution",
                CampaignRecordKindV1::SyntheticFixture => "synthetic_fixture",
            }),
        ),
        field("campaign_id", text(&record.campaign_id)),
        field(
            "provenance",
            Json::Object(vec![
                field("git_head", text(&record.provenance.git_head)),
                field(
                    "worktree_clean",
                    Json::Bool(record.provenance.worktree_clean),
                ),
                field("ci_run_id", json::unsigned(record.provenance.ci_run_id)),
                field(
                    "ci_green_on_exact_head",
                    Json::Bool(record.provenance.ci_green_on_exact_head),
                ),
            ]),
        ),
        field(
            "hardware",
            Json::Object(vec![
                field("backend", text(backend_name(record.hardware.backend))),
                field("device_name", text(&record.hardware.device_name)),
                field("driver_version", text(&record.hardware.driver_version)),
                field("host_os", text(&record.hardware.host_os)),
            ]),
        ),
        field("cells", Json::Array(cells)),
    ]);
    Ok(render(&value))
}

/// Parse an evidence record without validating it against a preregistration.
pub fn evidence_from_json(input: &str) -> Result<Dsv41CampaignEvidenceV1, Dsv41JsonError> {
    let mut root = Object::root(input, DSV41_EVIDENCE_KIND)?;
    let kind = match root.string("record_kind")?.as_str() {
        "physical_execution" => CampaignRecordKindV1::PhysicalExecution,
        "synthetic_fixture" => CampaignRecordKindV1::SyntheticFixture,
        _ => {
            return Err(Dsv41JsonError::UnknownVariant {
                field: "record_kind",
            })
        }
    };
    let campaign_id = root.string("campaign_id")?;
    let mut provenance_object = root.object("provenance")?;
    let provenance = CampaignProvenanceV1 {
        git_head: provenance_object.string("git_head")?,
        worktree_clean: provenance_object.bool("worktree_clean")?,
        ci_run_id: provenance_object.u64("ci_run_id")?,
        ci_green_on_exact_head: provenance_object.bool("ci_green_on_exact_head")?,
    };
    provenance_object.finish()?;
    let mut hardware_object = root.object("hardware")?;
    let hardware = CampaignHardwareV1 {
        backend: parse_backend("hardware.backend", &hardware_object.string("backend")?)?,
        device_name: hardware_object.string("device_name")?,
        driver_version: hardware_object.string("driver_version")?,
        host_os: hardware_object.string("host_os")?,
    };
    hardware_object.finish()?;
    let mut cells = Vec::new();
    for item in root.array("cells")? {
        let mut cell = Object::new("cells[]", item)?;
        let arm_id = cell.string("arm_id")?;
        let context_length = cell.u32("context_length")?;
        let decode_length = cell.u32("decode_length")?;
        let repetitions = cell.u32("repetitions")?;
        let quality = cell.nullable_f64("quality")?;
        let memory_metric = match cell.nullable("memory_metric")? {
            None => None,
            Some(value) => Some(parse_memory_metric(
                "memory_metric",
                &as_string("memory_metric", value)?,
            )?),
        };
        let memory_bytes = match cell.nullable("memory_bytes")? {
            None => None,
            Some(value) => Some(as_u64("memory_bytes", &value)?),
        };
        let decode_latency = match cell.nullable("decode_latency")? {
            None => None,
            Some(value) => {
                let mut latency = Object::new("decode_latency", value)?;
                let parsed = DecodeLatencyDistributionV1 {
                    samples: latency.u64("samples")?,
                    p50_ms: latency.f64("p50_ms")?,
                    p90_ms: latency.f64("p90_ms")?,
                    p99_ms: latency.f64("p99_ms")?,
                    max_ms: latency.f64("max_ms")?,
                };
                latency.finish()?;
                Some(parsed)
            }
        };
        let tokens_per_second = cell.nullable_f64("tokens_per_second")?;
        cell.finish()?;
        cells.push(CampaignCellV1 {
            arm_id,
            context_length,
            decode_length,
            repetitions,
            quality,
            memory_metric,
            memory_bytes,
            decode_latency,
            tokens_per_second,
        });
    }
    root.finish()?;
    Ok(Dsv41CampaignEvidenceV1 {
        kind,
        campaign_id,
        provenance,
        hardware,
        cells,
    })
}

/// Fail-closed JSON record errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dsv41JsonError {
    /// Input was not valid JSON.
    Syntax { offset: usize, reason: &'static str },
    /// `kind` tag did not match the expected record kind.
    WrongKind,
    /// `schema_version` was not the supported version.
    UnsupportedSchemaVersion,
    /// A required field was absent.
    MissingField { field: &'static str },
    /// An object contained a field not defined by the schema.
    UnknownField { field: String },
    /// A field had the wrong JSON type.
    WrongType { field: &'static str },
    /// An integer field was fractional, negative, or out of range.
    InvalidInteger { field: &'static str },
    /// A float field was not a finite number.
    InvalidNumber { field: &'static str },
    /// An enumerated string had an unknown value.
    UnknownVariant { field: &'static str },
    /// Parsed values violated the preregistration contract.
    Contract(Dsv41CampaignError),
}

impl From<Dsv41CampaignError> for Dsv41JsonError {
    fn from(error: Dsv41CampaignError) -> Self {
        Self::Contract(error)
    }
}

impl fmt::Display for Dsv41JsonError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax { offset, reason } => {
                write!(output, "JSON syntax error at byte {offset}: {reason}")
            }
            Self::WrongKind => output.write_str("record kind tag does not match"),
            Self::UnsupportedSchemaVersion => write!(
                output,
                "unsupported schema_version, expected {DSV41_CAMPAIGN_JSON_SCHEMA_VERSION}"
            ),
            Self::MissingField { field } => write!(output, "missing field {field}"),
            Self::UnknownField { field } => write!(output, "unknown field {field}"),
            Self::WrongType { field } => write!(output, "field {field} has the wrong type"),
            Self::InvalidInteger { field } => {
                write!(output, "field {field} is not a valid integer")
            }
            Self::InvalidNumber { field } => write!(output, "field {field} is not a finite number"),
            Self::UnknownVariant { field } => write!(output, "field {field} has an unknown value"),
            Self::Contract(error) => write!(output, "record contract: {error}"),
        }
    }
}

impl std::error::Error for Dsv41JsonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            _ => None,
        }
    }
}

struct Object {
    members: Vec<(String, Json)>,
}

impl Object {
    fn root(input: &str, kind: &str) -> Result<Self, Dsv41JsonError> {
        let value = json::parse(input).map_err(|error| Dsv41JsonError::Syntax {
            offset: error.offset,
            reason: error.reason,
        })?;
        let mut root = Self::new("root", value)?;
        if root.string("kind")? != kind {
            return Err(Dsv41JsonError::WrongKind);
        }
        let version = root.take("schema_version")?;
        if as_u64("schema_version", &version).ok()
            != Some(u64::from(DSV41_CAMPAIGN_JSON_SCHEMA_VERSION))
        {
            return Err(Dsv41JsonError::UnsupportedSchemaVersion);
        }
        Ok(root)
    }

    fn new(field: &'static str, value: Json) -> Result<Self, Dsv41JsonError> {
        match value {
            Json::Object(members) => Ok(Self { members }),
            _ => Err(Dsv41JsonError::WrongType { field }),
        }
    }

    fn take(&mut self, field: &'static str) -> Result<Json, Dsv41JsonError> {
        let index = self
            .members
            .iter()
            .position(|(key, _)| key == field)
            .ok_or(Dsv41JsonError::MissingField { field })?;
        Ok(self.members.remove(index).1)
    }

    fn nullable(&mut self, field: &'static str) -> Result<Option<Json>, Dsv41JsonError> {
        match self.take(field)? {
            Json::Null => Ok(None),
            value => Ok(Some(value)),
        }
    }

    fn string(&mut self, field: &'static str) -> Result<String, Dsv41JsonError> {
        as_string(field, self.take(field)?)
    }

    fn bool(&mut self, field: &'static str) -> Result<bool, Dsv41JsonError> {
        match self.take(field)? {
            Json::Bool(value) => Ok(value),
            _ => Err(Dsv41JsonError::WrongType { field }),
        }
    }

    fn u64(&mut self, field: &'static str) -> Result<u64, Dsv41JsonError> {
        as_u64(field, &self.take(field)?)
    }

    fn u32(&mut self, field: &'static str) -> Result<u32, Dsv41JsonError> {
        u32::try_from(self.u64(field)?).map_err(|_| Dsv41JsonError::InvalidInteger { field })
    }

    fn f64(&mut self, field: &'static str) -> Result<f64, Dsv41JsonError> {
        as_f64(field, &self.take(field)?)
    }

    fn nullable_f64(&mut self, field: &'static str) -> Result<Option<f64>, Dsv41JsonError> {
        match self.nullable(field)? {
            None => Ok(None),
            Some(value) => Ok(Some(as_f64(field, &value)?)),
        }
    }

    fn object(&mut self, field: &'static str) -> Result<Self, Dsv41JsonError> {
        Self::new(field, self.take(field)?)
    }

    fn array(&mut self, field: &'static str) -> Result<Vec<Json>, Dsv41JsonError> {
        match self.take(field)? {
            Json::Array(items) => Ok(items),
            _ => Err(Dsv41JsonError::WrongType { field }),
        }
    }

    fn u32_array(&mut self, field: &'static str) -> Result<Vec<u32>, Dsv41JsonError> {
        self.array(field)?
            .iter()
            .map(|value| {
                u32::try_from(as_u64(field, value)?)
                    .map_err(|_| Dsv41JsonError::InvalidInteger { field })
            })
            .collect()
    }

    fn finish(self) -> Result<(), Dsv41JsonError> {
        match self.members.into_iter().next() {
            None => Ok(()),
            Some((field, _)) => Err(Dsv41JsonError::UnknownField { field }),
        }
    }
}

fn as_string(field: &'static str, value: Json) -> Result<String, Dsv41JsonError> {
    match value {
        Json::String(text) => Ok(text),
        _ => Err(Dsv41JsonError::WrongType { field }),
    }
}

fn as_u64(field: &'static str, value: &Json) -> Result<u64, Dsv41JsonError> {
    match value {
        Json::Number(text) => {
            if !text.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(Dsv41JsonError::InvalidInteger { field });
            }
            text.parse::<u64>()
                .map_err(|_| Dsv41JsonError::InvalidInteger { field })
        }
        _ => Err(Dsv41JsonError::WrongType { field }),
    }
}

fn as_f64(field: &'static str, value: &Json) -> Result<f64, Dsv41JsonError> {
    match value {
        Json::Number(text) => {
            let parsed = text
                .parse::<f64>()
                .map_err(|_| Dsv41JsonError::InvalidNumber { field })?;
            if parsed.is_finite() {
                Ok(parsed)
            } else {
                Err(Dsv41JsonError::InvalidNumber { field })
            }
        }
        _ => Err(Dsv41JsonError::WrongType { field }),
    }
}

fn field(name: &str, value: Json) -> (String, Json) {
    (name.to_owned(), value)
}

fn text(value: &str) -> Json {
    Json::String(value.to_owned())
}

fn finite(field: &'static str, value: f64) -> Result<Json, Dsv41JsonError> {
    json::float(value).ok_or(Dsv41JsonError::InvalidNumber { field })
}

fn optional_finite(field: &'static str, value: Option<f64>) -> Result<Json, Dsv41JsonError> {
    value.map_or(Ok(Json::Null), |value| finite(field, value))
}

fn u32_array(values: &[u32]) -> Json {
    Json::Array(
        values
            .iter()
            .map(|&value| json::unsigned(u64::from(value)))
            .collect(),
    )
}

fn render(value: &Json) -> String {
    let mut output = String::new();
    json::write(value, &mut output);
    output
}

fn backend_name(backend: CampaignBackendV1) -> &'static str {
    match backend {
        CampaignBackendV1::Cpu => "cpu",
        CampaignBackendV1::Wgpu => "wgpu",
        CampaignBackendV1::LegacyCudaCrossCheck => "legacy_cuda_cross_check",
    }
}

fn parse_backend(field: &'static str, value: &str) -> Result<CampaignBackendV1, Dsv41JsonError> {
    match value {
        "cpu" => Ok(CampaignBackendV1::Cpu),
        "wgpu" => Ok(CampaignBackendV1::Wgpu),
        "legacy_cuda_cross_check" => Ok(CampaignBackendV1::LegacyCudaCrossCheck),
        _ => Err(Dsv41JsonError::UnknownVariant { field }),
    }
}

fn primitive_name(primitive: Dsv41PrimitiveV1) -> &'static str {
    match primitive {
        Dsv41PrimitiveV1::BoundedReplay => "bounded_replay",
        Dsv41PrimitiveV1::CrossLayerKvReuse => "cross_layer_kv_reuse",
        Dsv41PrimitiveV1::Fp4E2M1Kv => "fp4_e2m1_kv",
        Dsv41PrimitiveV1::SpeculativeVerification => "speculative_verification",
    }
}

fn parse_primitive(value: &str) -> Result<Dsv41PrimitiveV1, Dsv41JsonError> {
    match value {
        "bounded_replay" => Ok(Dsv41PrimitiveV1::BoundedReplay),
        "cross_layer_kv_reuse" => Ok(Dsv41PrimitiveV1::CrossLayerKvReuse),
        "fp4_e2m1_kv" => Ok(Dsv41PrimitiveV1::Fp4E2M1Kv),
        "speculative_verification" => Ok(Dsv41PrimitiveV1::SpeculativeVerification),
        _ => Err(Dsv41JsonError::UnknownVariant {
            field: "primitives[]",
        }),
    }
}

fn memory_metric_name(metric: CampaignMemoryMetricV1) -> &'static str {
    match metric {
        CampaignMemoryMetricV1::PeakProcessRssBytes => "peak_process_rss_bytes",
        CampaignMemoryMetricV1::PeakDeviceAllocatedBytes => "peak_device_allocated_bytes",
        CampaignMemoryMetricV1::PeakNnisOwnedAllocationBytes => "peak_nnis_owned_allocation_bytes",
    }
}

fn parse_memory_metric(
    field: &'static str,
    value: &str,
) -> Result<CampaignMemoryMetricV1, Dsv41JsonError> {
    match value {
        "peak_process_rss_bytes" => Ok(CampaignMemoryMetricV1::PeakProcessRssBytes),
        "peak_device_allocated_bytes" => Ok(CampaignMemoryMetricV1::PeakDeviceAllocatedBytes),
        "peak_nnis_owned_allocation_bytes" => {
            Ok(CampaignMemoryMetricV1::PeakNnisOwnedAllocationBytes)
        }
        _ => Err(Dsv41JsonError::UnknownVariant { field }),
    }
}

fn direction_name(direction: QualityDirectionV1) -> &'static str {
    match direction {
        QualityDirectionV1::HigherIsBetter => "higher_is_better",
        QualityDirectionV1::LowerIsBetter => "lower_is_better",
    }
}

fn parse_direction(value: &str) -> Result<QualityDirectionV1, Dsv41JsonError> {
    match value {
        "higher_is_better" => Ok(QualityDirectionV1::HigherIsBetter),
        "lower_is_better" => Ok(QualityDirectionV1::LowerIsBetter),
        _ => Err(Dsv41JsonError::UnknownVariant { field: "direction" }),
    }
}

#[cfg(test)]
mod tests {
    //! All values are synthetic placeholders; none is a measurement.

    use super::*;
    use crate::dsv41_campaign::{validate_dsv41_evidence, validate_dsv41_evidence_structure};

    const HEAD: &str = "0123456789abcdef0123456789abcdef01234567";
    const SHA: &str = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    fn prereg() -> Dsv41CampaignPreregistrationV1 {
        Dsv41CampaignPreregistrationV1::new(Dsv41CampaignPreregistrationFieldsV1 {
            campaign_id: "synthetic.dsv41-5.json-fixture".into(),
            model: CampaignModelV1 {
                model_id: "synthetic/model".into(),
                revision: HEAD.into(),
                weights_sha256: SHA.into(),
            },
            prompt_suite: CampaignPromptSuiteV1 {
                suite_id: "synthetic \"prompts\" é".into(),
                suite_sha256: SHA.into(),
                prompt_count: 2,
            },
            backend: CampaignBackendV1::Wgpu,
            arms: vec![
                CampaignArmV1 {
                    arm_id: "dense".into(),
                    primitives: Vec::new(),
                },
                CampaignArmV1 {
                    arm_id: "combo".into(),
                    primitives: vec![
                        Dsv41PrimitiveV1::BoundedReplay,
                        Dsv41PrimitiveV1::SpeculativeVerification,
                    ],
                },
            ],
            context_lengths: vec![128],
            decode_lengths: vec![8, 16],
            repetitions: 2,
            memory_metric: CampaignMemoryMetricV1::PeakDeviceAllocatedBytes,
            quality_metric: CampaignQualityMetricV1 {
                metric_id: "synthetic.quality".into(),
                direction: QualityDirectionV1::LowerIsBetter,
                max_regression: 0.1,
            },
        })
        .unwrap()
    }

    fn cell(arm_id: &str, decode_length: u32) -> CampaignCellV1 {
        CampaignCellV1 {
            arm_id: arm_id.into(),
            context_length: 128,
            decode_length,
            repetitions: 2,
            quality: Some(0.5),
            memory_metric: Some(CampaignMemoryMetricV1::PeakDeviceAllocatedBytes),
            memory_bytes: Some(u64::MAX),
            decode_latency: Some(DecodeLatencyDistributionV1 {
                samples: 2,
                p50_ms: 0.25,
                p90_ms: 0.5,
                p99_ms: 0.75,
                max_ms: 1.0,
            }),
            tokens_per_second: Some(1.5),
        }
    }

    fn evidence() -> Dsv41CampaignEvidenceV1 {
        Dsv41CampaignEvidenceV1 {
            kind: CampaignRecordKindV1::SyntheticFixture,
            campaign_id: "synthetic.dsv41-5.json-fixture".into(),
            provenance: CampaignProvenanceV1 {
                git_head: HEAD.into(),
                worktree_clean: true,
                ci_run_id: u64::MAX,
                ci_green_on_exact_head: true,
            },
            hardware: CampaignHardwareV1 {
                backend: CampaignBackendV1::Wgpu,
                device_name: "synthetic-device".into(),
                driver_version: "synthetic-driver".into(),
                host_os: "synthetic-os".into(),
            },
            cells: vec![
                cell("dense", 8),
                cell("dense", 16),
                cell("combo", 8),
                cell("combo", 16),
            ],
        }
    }

    #[test]
    fn preregistration_round_trips_exactly() {
        let record = prereg();
        let encoded = preregistration_to_json(&record).unwrap();
        assert!(encoded
            .starts_with(r#"{"kind":"nnis.dsv41.campaign-preregistration","schema_version":1,"#));
        let decoded = preregistration_from_json(&encoded).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(preregistration_to_json(&decoded).unwrap(), encoded);
    }

    #[test]
    fn evidence_round_trips_and_synthetic_stays_non_evidence() {
        let record = evidence();
        let encoded = evidence_to_json(&record).unwrap();
        let decoded = evidence_from_json(&encoded).unwrap();
        assert_eq!(decoded, record);
        assert_eq!(evidence_to_json(&decoded).unwrap(), encoded);
        validate_dsv41_evidence_structure(&prereg(), &decoded).unwrap();
        assert_eq!(
            validate_dsv41_evidence(&prereg(), &decoded),
            Err(Dsv41CampaignError::SyntheticFixtureIsNotEvidence)
        );
    }

    #[test]
    fn null_metrics_round_trip_and_are_rejected_by_validation() {
        let mut record = evidence();
        record.cells[1].decode_latency = None;
        record.cells[1].memory_metric = None;
        record.cells[1].memory_bytes = None;
        let decoded = evidence_from_json(&evidence_to_json(&record).unwrap()).unwrap();
        assert_eq!(decoded, record);
        assert!(matches!(
            validate_dsv41_evidence_structure(&prereg(), &decoded),
            Err(Dsv41CampaignError::MissingMetric { .. })
        ));
    }

    #[test]
    fn unknown_missing_version_and_kind_fail_closed() {
        let encoded = evidence_to_json(&evidence()).unwrap();
        assert_eq!(
            evidence_from_json(&encoded.replacen("\"cells\":", "\"extra\":1,\"cells\":", 1)),
            Err(Dsv41JsonError::UnknownField {
                field: "extra".into()
            })
        );
        assert_eq!(
            evidence_from_json(&encoded.replacen("\"host_os\"", "\"host_os\":\"x\",\"gpu\"", 1)),
            Err(Dsv41JsonError::UnknownField {
                field: "gpu".into()
            })
        );
        assert_eq!(
            evidence_from_json(&encoded.replacen(",\"tokens_per_second\":1.5", "", 1)),
            Err(Dsv41JsonError::MissingField {
                field: "tokens_per_second"
            })
        );
        assert_eq!(
            evidence_from_json(&encoded.replacen(
                "\"schema_version\":1",
                "\"schema_version\":2",
                1
            )),
            Err(Dsv41JsonError::UnsupportedSchemaVersion)
        );
        assert_eq!(
            evidence_from_json(&encoded.replacen(
                "\"schema_version\":1",
                "\"schema_version\":1.0",
                1
            )),
            Err(Dsv41JsonError::UnsupportedSchemaVersion)
        );
        assert_eq!(
            preregistration_from_json(&encoded),
            Err(Dsv41JsonError::WrongKind)
        );
        let prereg_json = preregistration_to_json(&prereg()).unwrap();
        assert_eq!(
            evidence_from_json(&prereg_json),
            Err(Dsv41JsonError::WrongKind)
        );
        assert!(matches!(
            preregistration_from_json(&prereg_json.replacen(
                "\"model\":",
                "\"kind\":\"x\",\"model\":",
                1
            )),
            Err(Dsv41JsonError::Syntax {
                reason: "duplicate object key",
                ..
            })
        ));
    }

    #[test]
    fn wrong_types_numbers_and_variants_fail_closed() {
        let encoded = evidence_to_json(&evidence()).unwrap();
        for (from, to, expected) in [
            (
                "\"worktree_clean\":true",
                "\"worktree_clean\":1",
                Dsv41JsonError::WrongType {
                    field: "worktree_clean",
                },
            ),
            (
                "\"repetitions\":2",
                "\"repetitions\":2.0",
                Dsv41JsonError::InvalidInteger {
                    field: "repetitions",
                },
            ),
            (
                "\"repetitions\":2",
                "\"repetitions\":-2",
                Dsv41JsonError::InvalidInteger {
                    field: "repetitions",
                },
            ),
            (
                "\"repetitions\":2",
                "\"repetitions\":4294967296",
                Dsv41JsonError::InvalidInteger {
                    field: "repetitions",
                },
            ),
            (
                "\"quality\":0.5",
                "\"quality\":1e999",
                Dsv41JsonError::InvalidNumber { field: "quality" },
            ),
            (
                "\"quality\":0.5",
                "\"quality\":\"0.5\"",
                Dsv41JsonError::WrongType { field: "quality" },
            ),
            (
                "\"record_kind\":\"synthetic_fixture\"",
                "\"record_kind\":\"measured\"",
                Dsv41JsonError::UnknownVariant {
                    field: "record_kind",
                },
            ),
            (
                "\"backend\":\"wgpu\"",
                "\"backend\":\"cuda\"",
                Dsv41JsonError::UnknownVariant {
                    field: "hardware.backend",
                },
            ),
        ] {
            assert_eq!(
                evidence_from_json(&encoded.replacen(from, to, 1)),
                Err(expected),
                "{to}"
            );
        }
        assert!(matches!(
            evidence_from_json("{\"kind\":"),
            Err(Dsv41JsonError::Syntax { .. })
        ));
        let prereg_json = preregistration_to_json(&prereg()).unwrap();
        assert_eq!(
            preregistration_from_json(&prereg_json.replacen(
                "\"repetitions\":2",
                "\"repetitions\":1",
                1
            )),
            Err(Dsv41JsonError::Contract(
                Dsv41CampaignError::TooFewRepetitions { repetitions: 1 }
            ))
        );
        assert_eq!(
            preregistration_from_json(&prereg_json.replacen("\"bounded_replay\"", "\"magic\"", 1)),
            Err(Dsv41JsonError::UnknownVariant {
                field: "primitives[]"
            })
        );
    }

    #[test]
    fn non_finite_values_cannot_be_encoded() {
        let mut record = evidence();
        record.cells[0].tokens_per_second = Some(f64::NAN);
        assert_eq!(
            evidence_to_json(&record),
            Err(Dsv41JsonError::InvalidNumber {
                field: "tokens_per_second"
            })
        );
    }
}
