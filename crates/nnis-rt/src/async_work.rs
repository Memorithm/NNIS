//! Ownership guards for asynchronous CUDA work.
//!
//! CUDA launches and copies may outlive the Rust call that submitted them.
//! Borrowed buffers therefore cannot be released until the device has passed
//! the last operation that can dereference their addresses. [`PendingGpuWork`]
//! couples an event with an arbitrary owned resource graph and keeps that graph
//! alive until the event completes.

use crate::{Event, Result, Stream};

/// Owned resources retained until a recorded stream tail has completed.
///
/// The type is intentionally generic: higher layers can retain device buffers,
/// modules, temporary host storage, or whole operation records as one resource
/// graph. Dropping an unfinished value waits for the completion event before
/// releasing the resources. If CUDA reports an error while that final wait is
/// attempted, the resource graph is leaked rather than risking a use-after-free
/// on the device.
#[derive(Debug)]
pub struct PendingGpuWork<R> {
    completion: Event,
    stream: Stream,
    resources: Option<R>,
}

impl<R> PendingGpuWork<R> {
    /// Record the current tail of `stream` and retain `resources` until it has
    /// completed.
    ///
    /// # Safety
    ///
    /// `resources` must own **every** Rust/CUDA object whose lifetime is relied
    /// upon by work already submitted to `stream` and not otherwise guaranteed
    /// to outlive that work. In particular, raw device addresses captured by a
    /// kernel or DMA operation must refer to allocations retained either here
    /// or by another owner whose lifetime is independently guaranteed.
    ///
    /// This constructor is unsafe because Rust cannot inspect previously
    /// submitted CUDA work to prove that the supplied ownership graph is
    /// complete. Safe high-level NNIS operations should encapsulate this call.
    pub unsafe fn from_enqueued(stream: &Stream, resources: R) -> Result<Self> {
        let completion = match Event::new(stream.ctx()) {
            Ok(event) => event,
            Err(error) => {
                Self::drain_or_leak(stream, resources);
                return Err(error);
            }
        };
        if let Err(error) = completion.record(stream) {
            Self::drain_or_leak(stream, resources);
            return Err(error);
        }
        Ok(Self {
            completion,
            stream: stream.clone(),
            resources: Some(resources),
        })
    }

    /// Return whether the retained stream tail has completed, without blocking.
    pub fn query(&self) -> Result<bool> {
        self.completion.query()
    }

    /// Wait for completion and return the retained resource graph.
    pub fn wait(mut self) -> Result<R> {
        self.completion.synchronize()?;
        Ok(self
            .resources
            .take()
            .expect("pending GPU work always owns its resources until completion"))
    }

    /// Stream on which the completion event was recorded.
    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Completion event marking the retained stream tail.
    pub fn completion_event(&self) -> &Event {
        &self.completion
    }

    fn drain_or_leak(stream: &Stream, resources: R) {
        if stream.synchronize().is_err() {
            // CUDA did not prove that the submitted work stopped using the
            // ownership graph. Leaking is preferable to freeing memory that a
            // still-running device operation may dereference.
            std::mem::forget(resources);
        }
    }
}

impl<R> Drop for PendingGpuWork<R> {
    fn drop(&mut self) {
        let Some(resources) = self.resources.take() else {
            return;
        };
        if self.completion.synchronize().is_err() {
            // Preserve memory safety if CUDA cannot establish completion.
            std::mem::forget(resources);
        }
    }
}
