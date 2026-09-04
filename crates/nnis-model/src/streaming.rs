use crate::sampling::HostLogitSampler;
use crate::{GenerationConfig, InferenceSession, SamplingConfig};
use nnis_rt::{NnisError, Result};

/// Caller control returned after each emitted token in a streaming generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationStreamControl {
    /// Continue generating until EOS or `max_new_tokens`.
    Continue,
    /// Stop cleanly after the current emitted token has been executed.
    Stop,
}

impl<'model> InferenceSession<'model> {
    /// Host-visible sampled generation with token-by-token delivery.
    ///
    /// The callback runs only after the emitted token has been executed through
    /// `decode_one`, so the session position and KV cache are continuation-ready
    /// even when the caller returns [`GenerationStreamControl::Stop`].
    ///
    /// This API intentionally preserves NNML1 sampling semantics: full logits
    /// are materialized on the host for sampling and the selected token is sent
    /// back to CUDA for decoder execution. Streaming therefore makes no NNML2
    /// device-residency or serving-performance claim.
    pub fn generate_sampled_streaming<F>(
        &mut self,
        input_ids: &[u32],
        generation: GenerationConfig,
        sampling: SamplingConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> GenerationStreamControl,
    {
        let required_positions = input_ids
            .len()
            .checked_add(generation.max_new_tokens)
            .ok_or_else(|| {
                NnisError::invalid_input(
                    "prompt + sampled streaming generation length overflows usize",
                )
            })?;
        if required_positions > self.capacity() {
            return Err(NnisError::invalid_input(format!(
                "prompt + sampled streaming generation requires {required_positions} positions; session capacity is {}",
                self.capacity()
            )));
        }

        let mut logits = self.prefill(input_ids)?;
        generation.validate(logits.len())?;
        let mut sampler = HostLogitSampler::new(sampling, logits.len())?;
        let mut generated = Vec::with_capacity(generation.max_new_tokens);

        for _ in 0..generation.max_new_tokens {
            let token = sampler.sample(&logits)?;
            generated.push(token);

            // Execute every emitted token before exposing it to the callback so
            // a clean caller stop leaves the KV/session state continuation-ready.
            logits = self.decode_one(token)?;

            let caller_control = on_token(token);
            if generation.eos_token_id == Some(token)
                || caller_control == GenerationStreamControl::Stop
            {
                break;
            }
        }

        Ok(generated)
    }
}
