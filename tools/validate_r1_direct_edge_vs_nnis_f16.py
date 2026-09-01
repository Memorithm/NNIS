#!/usr/bin/env python3
import argparse
import copy
import json
import math
import statistics
import sys
from pathlib import Path

EXPECTED_EDGE = "71dd1bae032e70771265917ec74d3ff4cad07a10"
EXPECTED_NNIS = "f00f3945be454dd6fcd0296a50de3d483c618884"
EXPECTED_MODEL = "80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1"
EXPECTED_TOKENIZER = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
EXPECTED_IDS = [260,3075,338,6650,260,2591,284,260,8872,1592,30,198,198,504,8872,314,253,8304,282,260,2591,30,657,314,253,19284,1248,338,21837,260,2591,30]

class EvidenceError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise EvidenceError(message)

def close(a, b, rel=1e-12, abs_tol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)

def validate(d):
    require(d.get("schema_version") == 1, "schema_version must be 1")
    require(d.get("kind") == "nnis_direct_cross_runtime_evidence", "unexpected kind")

    c = d["campaign"]
    require(c["rounds"] == 3 and c["processes_total"] == 12 and c["processes_per_runtime"] == 6, "campaign shape drift")
    require(c["order_per_round"] == ["Edge", "NNIS", "NNIS", "Edge"], "ABBA order drift")
    require(c["warmups_per_process"] == 2 and c["measured_requests_per_process"] == 1, "run-count drift")

    h = d["hardware"]
    require(h["gpu_name"] == "NVIDIA Thor" and h["compute_capability"] == "11.0", "hardware drift")
    require(h["power_mode"] == "MAXN", "power mode drift")
    require(h["gpu_gpc_hz"] == 1575000000 and h["gpu_nvd_hz"] == 1692000000 and h["emc_hz"] == 4266000000, "clock drift")
    require(h["competing_cuda_processes"] == 0 and h["regime_preserved_before_between_and_after_processes"] is True, "physical regime not qualified")

    m = d["model_identity"]
    require(m["source_model_sha256"] == EXPECTED_MODEL, "model SHA drift")
    require(m["tokenizer_sha256"] == EXPECTED_TOKENIZER, "tokenizer SHA drift")

    w = d["workload"]
    require(w["prompt_ids"] == [22007, 6463, 314] and w["generated_tokens"] == 32 and w["generation_forward_runs"] == 31, "workload drift")
    require(w["expected_generated_ids"] == EXPECTED_IDS, "trajectory drift")
    require(w["sampling_in_generation_stage"] is False and w["context_reuse"] is False, "metric workload drift")

    r = d["runtime_identity"]
    require(r["edge"]["source_commit"] == EXPECTED_EDGE, "Edge commit drift")
    require(r["nnis"]["source_commit"] == EXPECTED_NNIS, "NNIS commit drift")
    require(r["edge"]["dense_runtime_weight_dtype"] == "f16" and r["nnis"]["resident_weight_dtype"] == "f16", "declared F16 storage drift")
    require(r["nnis"]["projection_accumulator_dtype"] == "f32" and r["nnis"]["attention_accumulator_dtype"] == "f32", "NNIS accumulator contract drift")

    metric = d["metric_contract"]
    require(metric["common_between_runtimes"] is True, "metric must be common")

    s = d["semantic_gate"]
    require(all(s[k] is True for k in ["same_pinned_source_checkpoint", "same_tokenizer_identity_prequalified", "same_prompt", "same_batch_size", "same_decode_length", "declared_runtime_weight_storage_f16_both", "all_process_outputs_match_qualified_semantic_output", "exact_32_token_trajectory_prequalified"]), "semantic gate drift")
    require(s["numerical_or_bitwise_logit_equivalence_claimed"] is False, "must not claim numerical equivalence")

    raw = d["raw"]
    for key in raw:
        require(len(raw[key]) == 6, f"{key} must contain six process values")
    eg = raw["edge_generation_stage_gpu_ms"]
    ng = raw["nnis_generation_stage_gpu_ms"]
    et = raw["edge_generation_tps_common_definition"]
    nt = raw["nnis_generation_tps_common_definition"]
    ep = raw["edge_prefill_gpu_ms"]
    np = raw["nnis_prefill_gpu_ms"]
    for t, g in zip(et, eg): require(close(t, 32000.0/g, rel=2e-6), "Edge TPS formula drift")
    for t, g in zip(nt, ng): require(close(t, 32000.0/g, rel=1e-10), "NNIS TPS formula drift")

    a = d["aggregate"]
    require(close(a["edge_generation_stage_gpu_median_ms"], statistics.median(eg)), "Edge generation median drift")
    require(close(a["nnis_generation_stage_gpu_median_ms"], statistics.median(ng)), "NNIS generation median drift")
    require(close(a["nnis_over_edge_generation_latency_ratio"], statistics.median(ng)/statistics.median(eg)), "generation latency ratio drift")
    require(close(a["generation_separation_min_nnis_minus_max_edge_ms"], min(ng)-max(eg)), "generation separation drift")
    require(a["all_edge_generation_samples_below_all_nnis_generation_samples"] is (max(eg) < min(ng)), "distribution ordering drift")
    require(close(a["edge_generation_tps_median"], statistics.median(et)), "Edge TPS median drift")
    require(close(a["nnis_generation_tps_median"], statistics.median(nt)), "NNIS TPS median drift")
    require(close(a["edge_over_nnis_generation_throughput_ratio"], statistics.median(et)/statistics.median(nt)), "throughput ratio drift")
    require(close(a["edge_prefill_gpu_median_ms"], statistics.median(ep)), "Edge prefill median drift")
    require(close(a["nnis_prefill_gpu_median_ms"], statistics.median(np)), "NNIS prefill median drift")

    comp = d["comparability"]
    require(comp["operational_speed_ratio_allowed_for_this_specific_workload_and_metric"] is True, "qualified operational ratio unexpectedly disabled")
    require(comp["precision_semantic_equivalence_claimed"] is False, "precision-semantic equivalence must remain false")
    require(comp["general_performance_equivalence_claimed"] is False, "general equivalence must remain false")
    require(comp["cross_runtime_memory_ratio_allowed"] is False, "memory ratio must remain blocked")


def self_test(d):
    cases = []
    x = copy.deepcopy(d); x["comparability"]["precision_semantic_equivalence_claimed"] = True; cases.append(x)
    x = copy.deepcopy(d); x["comparability"]["cross_runtime_memory_ratio_allowed"] = True; cases.append(x)
    x = copy.deepcopy(d); x["runtime_identity"]["nnis"]["source_commit"] = "bad"; cases.append(x)
    x = copy.deepcopy(d); x["raw"]["nnis_generation_stage_gpu_ms"][0] += 10; cases.append(x)
    x = copy.deepcopy(d); x["workload"]["generation_forward_runs"] = 32; cases.append(x)
    for i, bad in enumerate(cases, 1):
        try:
            validate(bad)
        except EvidenceError:
            continue
        raise EvidenceError(f"negative self-test {i} unexpectedly passed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("path")
    args = p.parse_args()
    d = json.loads(Path(args.path).read_text())
    validate(d)
    if args.self_test:
        self_test(d)
    print("R1_DIRECT_EDGE_VS_NNIS_F16_EVIDENCE_OK")

if __name__ == "__main__":
    try:
        main()
    except (EvidenceError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"evidence validation failed: {e}", file=sys.stderr)
        sys.exit(1)
