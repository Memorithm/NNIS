use super::InferenceSession;
use nnis_rt::{compact_kv_cache, Result};

impl<'model> InferenceSession<'model> {
    /// Compact every model layer to the same explicit subset of the cache's
    /// currently active physical rows.
    ///
    /// The row indices address the current contiguous physical KV prefix, not
    /// logical token IDs and not RoPE positions. The selection must be strictly
    /// increasing, unique, and in range. Already-encoded K/V vectors retain
    /// their original RoPE phases when moved to the new physical prefix.
    ///
    /// This method first drains outstanding decoder work so pending append
    /// ownership can be retired safely. It then performs whole-cache
    /// replacement through `nnis-rt`; the session's logical `position` is not
    /// changed. Consequently, later decoded tokens keep using their original
    /// monotonically increasing logical/RoPE positions while attention scans
    /// only the retained physical history plus subsequent appends.
    ///
    /// Reducing active rows does not reduce the cache allocation capacity and
    /// carries no latency, bandwidth, HBM-residency, throughput, or quality
    /// claim by itself.
    pub fn compact_kv_cache_rows(&mut self, retained_rows: &[usize]) -> Result<()> {
        self.stream.synchronize()?;
        self.pending_appends.clear();

        let logical_position = self.position;
        compact_kv_cache(&mut self.cache, retained_rows)?;
        debug_assert_eq!(
            self.position, logical_position,
            "KV row compaction must not mutate logical decoder position"
        );
        Ok(())
    }
}
