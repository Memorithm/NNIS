# License provenance audit v1

This tool supports issue #150 by inventorying repository evidence before any
license change. It is intentionally **not** a relicensing tool.

Run:

```bash
python3 tools/audit_license_provenance.py --root . --output /tmp/nnis-license-provenance.json
```

The JSON report records the exact Git commit, dirty state, tracked files,
authors observed in Git history, license/notice files, Cargo `license` and
`license-file` metadata, SPDX/copyright markers, and vendored/generated or
external-marker paths that need review.

The report always sets `automatic_relicensing_ready=false`. A source-tree
scan cannot prove ownership of every contribution, contributor consent, or the
legal effect of historical license grants. Those questions require documented
rights evidence and, where appropriate, legal review.

No future change from the current repository license metadata should be
inferred from this audit alone. Third-party copyrights, notices, licenses and
attribution obligations must be preserved independently of the license chosen
for NNIS-owned code.
