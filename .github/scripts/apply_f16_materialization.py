from pathlib import Path

lib = Path("crates/nnis-model/src/lib.rs")
text = lib.read_text()
if "mod f16_materialization_memory;" not in text:
    text = text.replace(
        "mod f16_fused_projection_candidate;\n",
        "mod f16_fused_projection_candidate;\nmod f16_materialization_memory;\n",
        1,
    )
export_anchor = "pub use f16_fused_projection_candidate::F16FusedProjectionGroupsCandidate;\n"
export = (
    "pub use f16_materialization_memory::{\n"
    "    F16WeightMaterializationEventKindV1, F16WeightMaterializationEventV1,\n"
    "    F16WeightMaterializationMemoryEvidenceV1,\n"
    "    NNIS_F16_WEIGHT_MATERIALIZATION_MEMORY_EVIDENCE_VERSION,\n"
    "};\n"
)
if "F16WeightMaterializationMemoryEvidenceV1" not in text:
    if export_anchor not in text:
        raise SystemExit("lib export anchor missing")
    text = text.replace(export_anchor, export_anchor + export, 1)
lib.write_text(text)

path = Path("crates/nnis-model/src/f16_reference_runtime.rs")
text = path.read_text()
import_anchor = "use crate::runtime::build_rope_cache;\n"
import_line = (
    "use crate::f16_materialization_memory::{\n"
    "    F16WeightMaterializationMemoryEvidenceV1, F16WeightMaterializationTracker,\n"
    "};\n"
)
if "F16WeightMaterializationTracker" not in text:
    if import_anchor not in text:
        raise SystemExit("runtime import anchor missing")
    text = text.replace(import_anchor, import_line + import_anchor, 1)

start = text.index("    fn from_f32(")
end_marker = "\n}\n\n#[derive(Debug)]\npub struct F16ReferenceModel"
end = text.index(end_marker, start)
new_fn = r'''    fn from_f32(
        source: &ModelWeights,
        stream: &Stream,
        kernels: &F16ReferenceKernels,
        execution_plan: F16ReferenceExecutionPlan,
        projection_candidate: Option<&F16TransposedProjectionCandidate>,
        source_owned_allocation_bytes: u64,
    ) -> Result<(Self, F16WeightMaterializationTracker)> {
        fn buffer_bytes(name: &str, buffer: &DeviceBuffer<u16>) -> Result<u64> {
            u64::try_from(buffer.size_bytes()).map_err(|_| {
                NnisError::invalid_input(format!("F16 allocation {name} byte size exceeds u64"))
            })
        }

        fn narrow_resident(
            name: &str,
            stream: &Stream,
            kernels: &F16ReferenceKernels,
            source: &DeviceBuffer<f32>,
            tracker: &mut F16WeightMaterializationTracker,
        ) -> Result<Arc<DeviceBuffer<u16>>> {
            let output = Arc::new(DeviceBuffer::<u16>::new(stream.ctx(), source.len())?);
            tracker.allocate_resident(name, buffer_bytes(name, &output)?)?;
            unsafe { kernels.enqueue_narrow_from_f32(stream, source, &output)? };
            stream.synchronize()?;
            Ok(output)
        }

        fn narrow_temporary(
            name: &str,
            stream: &Stream,
            kernels: &F16ReferenceKernels,
            source: &DeviceBuffer<f32>,
            tracker: &mut F16WeightMaterializationTracker,
        ) -> Result<Arc<DeviceBuffer<u16>>> {
            let output = Arc::new(DeviceBuffer::<u16>::new(stream.ctx(), source.len())?);
            tracker.allocate_temporary(name, buffer_bytes(name, &output)?)?;
            unsafe { kernels.enqueue_narrow_from_f32(stream, source, &output)? };
            stream.synchronize()?;
            Ok(output)
        }

        fn narrow_projection(
            name: &str,
            stream: &Stream,
            kernels: &F16ReferenceKernels,
            source: &MatrixWeight,
            layout: F16ReferenceProjectionLayout,
            projection_candidate: Option<&F16TransposedProjectionCandidate>,
            tracker: &mut F16WeightMaterializationTracker,
        ) -> Result<Arc<DeviceBuffer<u16>>> {
            match layout {
                F16ReferenceProjectionLayout::KnReference => narrow_resident(
                    name,
                    stream,
                    kernels,
                    source.tensor().as_f32()?,
                    tracker,
                ),
                F16ReferenceProjectionLayout::NkTransposedCandidate
                | F16ReferenceProjectionLayout::NkTransposedFusedGroupsCandidate
                | F16ReferenceProjectionLayout::NkTransposedFusedMlpCandidate => {
                    let temporary_name = format!("{name}.kn_temporary");
                    let kn = narrow_temporary(
                        &temporary_name,
                        stream,
                        kernels,
                        source.tensor().as_f32()?,
                        tracker,
                    )?;
                    let kn_bytes = buffer_bytes(&temporary_name, &kn)?;
                    let nk = Arc::new(DeviceBuffer::<u16>::new(stream.ctx(), kn.len())?);
                    tracker.allocate_resident(name, buffer_bytes(name, &nk)?)?;
                    let candidate = projection_candidate.ok_or_else(|| {
                        NnisError::unsupported(
                            "F16 transposed projection plan selected without candidate kernels",
                        )
                    })?;
                    unsafe {
                        candidate.enqueue_transpose_kn_to_nk(
                            stream,
                            &kn,
                            &nk,
                            source.rows(),
                            source.cols(),
                        )?;
                    }
                    stream.synchronize()?;
                    tracker.release_temporary(&temporary_name, kn_bytes)?;
                    drop(kn);
                    Ok(nk)
                }
            }
        }

        let mut tracker = F16WeightMaterializationTracker::new(source_owned_allocation_bytes)?;
        let layout = execution_plan.projection_layout;
        let token_embedding = narrow_resident(
            "token_embedding",
            stream,
            kernels,
            source.token_embedding.tensor().as_f32()?,
            &mut tracker,
        )?;
        let mut layers = Vec::with_capacity(source.layers.len());
        for (index, layer) in source.layers.iter().enumerate() {
            layers.push(F16DecoderLayerWeights {
                input_norm: narrow_resident(
                    &format!("layers.{index}.input_norm"),
                    stream,
                    kernels,
                    layer.input_norm.tensor().as_f32()?,
                    &mut tracker,
                )?,
                q_proj: narrow_projection(
                    &format!("layers.{index}.q_proj"), stream, kernels, &layer.q_proj, layout,
                    projection_candidate, &mut tracker,
                )?,
                k_proj: narrow_projection(
                    &format!("layers.{index}.k_proj"), stream, kernels, &layer.k_proj, layout,
                    projection_candidate, &mut tracker,
                )?,
                v_proj: narrow_projection(
                    &format!("layers.{index}.v_proj"), stream, kernels, &layer.v_proj, layout,
                    projection_candidate, &mut tracker,
                )?,
                o_proj: narrow_projection(
                    &format!("layers.{index}.o_proj"), stream, kernels, &layer.o_proj, layout,
                    projection_candidate, &mut tracker,
                )?,
                post_attention_norm: narrow_resident(
                    &format!("layers.{index}.post_attention_norm"),
                    stream,
                    kernels,
                    layer.post_attention_norm.tensor().as_f32()?,
                    &mut tracker,
                )?,
                gate_proj: narrow_projection(
                    &format!("layers.{index}.gate_proj"), stream, kernels, &layer.gate_proj,
                    layout, projection_candidate, &mut tracker,
                )?,
                up_proj: narrow_projection(
                    &format!("layers.{index}.up_proj"), stream, kernels, &layer.up_proj, layout,
                    projection_candidate, &mut tracker,
                )?,
                down_proj: narrow_projection(
                    &format!("layers.{index}.down_proj"), stream, kernels, &layer.down_proj,
                    layout, projection_candidate, &mut tracker,
                )?,
            });
        }
        let final_norm = narrow_resident(
            "final_norm",
            stream,
            kernels,
            source.final_norm.tensor().as_f32()?,
            &mut tracker,
        )?;
        let lm_head = narrow_projection(
            "lm_head",
            stream,
            kernels,
            &source.lm_head,
            layout,
            projection_candidate,
            &mut tracker,
        )?;
        Ok((
            Self {
                token_embedding,
                layers,
                final_norm,
                lm_head,
            },
            tracker,
        ))
    }
'''
text = text[:start] + new_fn + text[end:]

field_anchor = "    weights: F16ModelWeights,\n"
field_line = "    materialization_memory_evidence: F16WeightMaterializationMemoryEvidenceV1,\n"
if field_line not in text:
    if field_anchor not in text:
        raise SystemExit("model field anchor missing")
    text = text.replace(field_anchor, field_anchor + field_line, 1)

old_build = '''        let resident_weights = F16ModelWeights::from_f32(
            &weights,
            stream,
            &kernels,
            execution_plan,
            projection_candidate.as_ref(),
        )?;
'''
new_build = '''        let source_weight_allocations = weights.weight_allocation_summary_v1()?;
        let (resident_weights, materialization_tracker) = F16ModelWeights::from_f32(
            &weights,
            stream,
            &kernels,
            execution_plan,
            projection_candidate.as_ref(),
            source_weight_allocations.owned_device_allocation_bytes,
        )?;
        let steady_state_f16_weight_allocations = resident_weights.weight_allocation_summary_v1()?;
        let materialization_memory_evidence = materialization_tracker.finish(
            execution_plan,
            source_weight_allocations,
            steady_state_f16_weight_allocations,
        )?;
'''
if old_build not in text:
    raise SystemExit("constructor build block missing")
text = text.replace(old_build, new_build, 1)

init_anchor = "            weights: resident_weights,\n"
init_line = "            materialization_memory_evidence,\n"
if init_line not in text:
    if init_anchor not in text:
        raise SystemExit("initializer anchor missing")
    text = text.replace(init_anchor, init_anchor + init_line, 1)

getter_anchor = '''    pub fn weight_allocation_summary_v1(&self) -> Result<WeightAllocationSummaryV1> {
        self.weights.weight_allocation_summary_v1()
    }

'''
getter = getter_anchor + '''    /// Exact successful-construction allocation-lifetime evidence for the F32 -> F16 weight materialization scope.
    pub fn weight_materialization_memory_evidence_v1(
        &self,
    ) -> &F16WeightMaterializationMemoryEvidenceV1 {
        &self.materialization_memory_evidence
    }

'''
if "pub fn weight_materialization_memory_evidence_v1" not in text:
    if getter_anchor not in text:
        raise SystemExit("getter anchor missing")
    text = text.replace(getter_anchor, getter, 1)
path.write_text(text)
