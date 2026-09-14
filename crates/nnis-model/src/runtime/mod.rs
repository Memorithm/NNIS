// Keep the established decoder runtime byte-for-byte in `runtime.rs` while
// allowing narrowly scoped extensions to live in child modules with access to
// `InferenceSession`'s private invariants.
include!("../runtime.rs");

mod kv_selection;
