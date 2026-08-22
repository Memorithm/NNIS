//! Reusable in-memory cache for NVRTC output.

use crate::program::{CodeKind, CompileOptions, JitProgram, ProgramCacheKey};
use nnis_rt::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// Immutable result of one successful NVRTC compilation.
#[derive(Debug)]
pub struct CompiledCode {
    bytes: Arc<[u8]>,
    kind: CodeKind,
    key: ProgramCacheKey,
    log: Arc<str>,
}

impl CompiledCode {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn shared_bytes(&self) -> Arc<[u8]> {
        Arc::clone(&self.bytes)
    }

    pub fn kind(&self) -> CodeKind {
        self.kind
    }

    pub fn key(&self) -> ProgramCacheKey {
        self.key
    }

    pub fn log(&self) -> &str {
        &self.log
    }
}

/// Process-local compiled-code cache. Compilation happens outside the mutex;
/// concurrent misses may compile twice, but only one immutable result is kept.
#[derive(Debug, Default)]
pub struct JitCompiler {
    cache: Mutex<HashMap<ProgramCacheKey, Arc<CompiledCode>>>,
}

impl JitCompiler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn compile_ptx(&self, source: &str, options: &CompileOptions) -> Result<Arc<CompiledCode>> {
        self.compile(source, options, CodeKind::Ptx)
    }

    pub fn compile_cubin(
        &self,
        source: &str,
        options: &CompileOptions,
    ) -> Result<Arc<CompiledCode>> {
        self.compile(source, options, CodeKind::Cubin)
    }

    pub fn compile(
        &self,
        source: &str,
        options: &CompileOptions,
        kind: CodeKind,
    ) -> Result<Arc<CompiledCode>> {
        let key = ProgramCacheKey::from_source(source, options, kind);
        if let Some(hit) = lock_unpoisoned(&self.cache).get(&key) {
            return Ok(Arc::clone(hit));
        }

        let program = JitProgram::compile(source, options.clone())?;
        let bytes = match kind {
            CodeKind::Ptx => program.ptx()?,
            CodeKind::Cubin => program.cubin()?,
        };
        let compiled = Arc::new(CompiledCode {
            bytes: bytes.into(),
            kind,
            key,
            log: Arc::from(program.log()),
        });

        let mut cache = lock_unpoisoned(&self.cache);
        Ok(Arc::clone(
            cache.entry(key).or_insert_with(|| Arc::clone(&compiled)),
        ))
    }

    pub fn len(&self) -> usize {
        lock_unpoisoned(&self.cache).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        lock_unpoisoned(&self.cache).clear();
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
