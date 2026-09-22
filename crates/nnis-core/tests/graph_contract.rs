use nnis_core::graph::{
    F32GraphLimitsV1, F32GraphV1, F32NodeV1, F32OpV1, F32ShapeV1, F32_GRAPH_POLICY,
    F32_GRAPH_VERSION,
};

const INPUTS: [F32ShapeV1; 2] = [
    F32ShapeV1::Vector(2),
    F32ShapeV1::Matrix { rows: 2, cols: 2 },
];
const NODES: [F32NodeV1<'static>; 2] = [
    F32NodeV1 {
        operation: F32OpV1::ProjectKn {
            input: 0,
            weights: 1,
        },
        output: F32ShapeV1::Vector(2),
    },
    F32NodeV1 {
        operation: F32OpV1::Relu { input: 2 },
        output: F32ShapeV1::Vector(2),
    },
];

fn limits() -> F32GraphLimitsV1 {
    F32GraphLimitsV1 {
        max_inputs: 8,
        max_nodes: 16,
        max_tensor_bytes: 4096,
        max_live_payload_bytes: 16384,
        max_scratch_bytes: 4096,
        max_work_items: 10000,
    }
}

fn plan<'a>(inputs: &'a [F32ShapeV1], nodes: &'a [F32NodeV1<'a>]) -> F32GraphV1<'a> {
    F32GraphV1 {
        schema_version: F32_GRAPH_VERSION,
        numerical_policy: F32_GRAPH_POLICY,
        inputs,
        nodes,
    }
}

#[test]
fn accounting_charges_all_values_and_largest_scratch() {
    let valid = plan(&INPUTS, &NODES).validate(limits()).unwrap();
    let budget = valid.budget();
    assert_eq!(budget.input_payload_bytes, 24);
    assert_eq!(budget.node_payload_bytes, 16);
    assert_eq!(budget.max_scratch_payload_bytes, 8);
    assert_eq!(budget.live_payload_bound_bytes, 48);
    assert_eq!(budget.work_items, 24);
    assert_eq!(valid.plan().nodes, &NODES);
}

#[test]
fn exact_limits_are_inclusive_and_each_smaller_limit_fails() {
    let exact = F32GraphLimitsV1 {
        max_inputs: 2,
        max_nodes: 2,
        max_tensor_bytes: 16,
        max_live_payload_bytes: 48,
        max_scratch_bytes: 8,
        max_work_items: 24,
    };
    assert!(plan(&INPUTS, &NODES).validate(exact).is_ok());
    for smaller in [
        F32GraphLimitsV1 {
            max_inputs: 1,
            ..exact
        },
        F32GraphLimitsV1 {
            max_nodes: 1,
            ..exact
        },
        F32GraphLimitsV1 {
            max_tensor_bytes: 15,
            ..exact
        },
        F32GraphLimitsV1 {
            max_live_payload_bytes: 47,
            ..exact
        },
        F32GraphLimitsV1 {
            max_scratch_bytes: 7,
            ..exact
        },
        F32GraphLimitsV1 {
            max_work_items: 23,
            ..exact
        },
    ] {
        assert!(plan(&INPUTS, &NODES).validate(smaller).is_err());
    }
}

#[test]
fn zero_limits_are_never_interpreted_as_unlimited() {
    let standard = limits();
    for zero in [
        F32GraphLimitsV1 {
            max_inputs: 0,
            ..standard
        },
        F32GraphLimitsV1 {
            max_nodes: 0,
            ..standard
        },
        F32GraphLimitsV1 {
            max_tensor_bytes: 0,
            ..standard
        },
        F32GraphLimitsV1 {
            max_live_payload_bytes: 0,
            ..standard
        },
        F32GraphLimitsV1 {
            max_scratch_bytes: 0,
            ..standard
        },
        F32GraphLimitsV1 {
            max_work_items: 0,
            ..standard
        },
    ] {
        assert!(plan(&INPUTS, &NODES).validate(zero).is_err());
    }
}

#[test]
fn public_literal_versions_and_policy_are_revalidated() {
    let original = plan(&INPUTS, &NODES);
    assert!(F32GraphV1 {
        schema_version: 2,
        ..original
    }
    .validate(limits())
    .is_err());
    assert!(F32GraphV1 {
        numerical_policy: "fast-math",
        ..original
    }
    .validate(limits())
    .is_err());
    assert!(plan(&[], &NODES).validate(limits()).is_err());
    assert!(plan(&INPUTS, &[]).validate(limits()).is_err());
}

#[test]
fn self_reference_future_reference_and_missing_operand_fail_closed() {
    for input in [2, 3, usize::MAX] {
        let mut nodes = NODES;
        nodes[0].operation = F32OpV1::Relu { input };
        assert!(plan(&INPUTS, &nodes).validate(limits()).is_err());
    }
}

#[test]
fn invalid_late_node_is_rejected_during_static_validation() {
    let mut nodes = NODES;
    nodes[1].operation = F32OpV1::Relu { input: 3 };
    assert!(plan(&INPUTS, &nodes).validate(limits()).is_err());
}

#[test]
fn matrix_orientation_is_not_inferred_from_equal_byte_length() {
    let mut inputs = INPUTS;
    inputs[1] = F32ShapeV1::Matrix { rows: 1, cols: 4 };
    assert!(plan(&inputs, &NODES).validate(limits()).is_err());
    inputs[1] = F32ShapeV1::Vector(4);
    assert!(plan(&inputs, &NODES).validate(limits()).is_err());
}

#[test]
fn output_shape_and_vector_matrix_kind_must_match_exactly() {
    for shape in [
        F32ShapeV1::Vector(1),
        F32ShapeV1::Matrix { rows: 1, cols: 2 },
    ] {
        let mut nodes = NODES;
        nodes[0].output = shape;
        assert!(plan(&INPUTS, &nodes).validate(limits()).is_err());
    }
}

#[test]
fn zero_dimensions_and_tensor_byte_overflow_are_errors() {
    for shape in [
        F32ShapeV1::Vector(0),
        F32ShapeV1::Vector(u64::MAX),
        F32ShapeV1::Matrix { rows: 0, cols: 2 },
        F32ShapeV1::Matrix {
            rows: u64::MAX,
            cols: 2,
        },
    ] {
        assert!(shape.bytes().is_err());
    }
}

#[test]
fn input_total_overflow_cannot_wrap_under_a_memory_limit() {
    let inputs = [F32ShapeV1::Vector(1_u64 << 60); 4];
    let nodes = [F32NodeV1 {
        operation: F32OpV1::Sum { input: 0 },
        output: F32ShapeV1::Vector(1),
    }];
    let large = F32GraphLimitsV1 {
        max_tensor_bytes: u64::MAX,
        max_live_payload_bytes: u64::MAX,
        max_work_items: u64::MAX,
        ..limits()
    };
    assert!(plan(&inputs, &nodes).validate(large).is_err());
}

#[test]
fn gather_checks_whole_index_list_and_accepts_duplicates() {
    let inputs = [F32ShapeV1::Vector(2)];
    let accepted = [1, 0, 1];
    let node = F32NodeV1 {
        operation: F32OpV1::Gather {
            input: 0,
            indices: &accepted,
        },
        output: F32ShapeV1::Vector(3),
    };
    assert!(plan(&inputs, &[node]).validate(limits()).is_ok());
    for indices in [&[0, 2, 0][..], &[usize::MAX][..], &[][..]] {
        let nodes = [F32NodeV1 {
            operation: F32OpV1::Gather { input: 0, indices },
            output: F32ShapeV1::Vector(indices.len() as u64),
        }];
        assert!(plan(&inputs, &nodes).validate(limits()).is_err());
    }
}

#[test]
fn scatter_binds_base_source_and_indices_without_in_place_output() {
    let inputs = [F32ShapeV1::Vector(2), F32ShapeV1::Vector(3)];
    let indices = [1, 1, 0];
    let mut nodes = [F32NodeV1 {
        operation: F32OpV1::ScatterAdd {
            base: 0,
            source: 1,
            indices: &indices,
        },
        output: F32ShapeV1::Vector(2),
    }];
    assert!(plan(&inputs, &nodes).validate(limits()).is_ok());
    nodes[0].operation = F32OpV1::ScatterAdd {
        base: 0,
        source: 1,
        indices: &[0],
    };
    assert!(plan(&inputs, &nodes).validate(limits()).is_err());
    nodes[0].operation = F32OpV1::ScatterAdd {
        base: 2,
        source: 1,
        indices: &indices,
    };
    assert!(plan(&inputs, &nodes).validate(limits()).is_err());
}
