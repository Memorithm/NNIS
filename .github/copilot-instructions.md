# NNIS repository agent instructions

Before making repository changes, fetch and read the persistent NNIS agent roadmap:

```bash
git fetch origin agent/sovereignty-roadmap && \
git show origin/agent/sovereignty-roadmap:.agent/NNIS_SOVEREIGNTY_ROADMAP.yaml
```

Treat `AGENTS.md` at repository root as mandatory bootstrap policy. Reread the roadmap at every session start, before a new major task, after benchmark promotion/rejection, after strategy changes, and before performance/representation PR or merge decisions.

If the roadmap is unavailable, fail closed for major architectural, performance-promotion, representation-format, or merge decisions. Do not substitute guesses for missing roadmap state.
