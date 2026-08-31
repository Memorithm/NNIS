# F32AttentionPlan v1

`F32AttentionPlan` is a versioned execution-policy axis for one-token cached
decoder attention. It is independent of projection selection, weight
representation and activation fusion.

Schema version 1 admits exactly two states:

- `serial_single_thread`: historical correctness-first cached attention;
- `parallel_value` with `threads_per_query_head = 64`: R2 candidate qualified on
  the physical SmolLM2/Thor sweep.

The runtime default remains `serial_single_thread`.

Existing `Model::new_with_all_plans` and
`Model::load_directory_with_all_plans` signatures are preserved and inject the
baseline attention plan. Opt-in callers use `new_with_execution_plans` or
`load_directory_with_execution_plans`.

The candidate path fails closed unless the model head dimension is exactly 64.
This prevents extrapolating physical evidence to unmeasured geometries.

The plan does not alter model format v1 or weight representation.
