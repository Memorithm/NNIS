# License provenance audit v1

This tool supports issue #150 by inventorying repository evidence before any
license change. It is intentionally **not** a relicensing tool.

Run:

```bash
python3 tools/audit_license_provenance.py \
  --root . \
  --policy policy/license-provenance-v1.json \
  --output /tmp/nnis-license-provenance.json
```

The JSON report records the exact Git commit, dirty state, tracked files,
authors observed in Git history, license/notice files, Cargo `license`,
`license-file`, and workspace-inheritance metadata, locked dependency identities,
SPDX/copyright markers, and vendored/generated or external-marker paths that
need review.

The versioned policy makes the reviewed repository state fail closed in CI.
It detects unreviewed changes to Cargo manifests, workspace license metadata,
license/notice files anywhere in the tracked tree, workspace inheritance, required evidence files, and
the exact `Cargo.lock` dependency baseline. An intentional dependency or
licensing-metadata change must update the policy in the same reviewed change.
All package and workspace-package `license` and `license-file` values, plus
inheritance flags, are parsed as TOML and compared exactly, including valid
single-quoted values.
Required evidence files are pinned by SHA-256, so changing the audit
implementation, workflow, or documentation also requires an explicit baseline
update.

License/notice families include `LICENSE`, `NOTICE`, `COPYING`, and `AUTHORS`,
with common extension, hyphen, and underscore suffixes. Every matching tracked
path and SHA-256 content identity is policy-checked, including nested third-party
files; the root-only fields remain as an explicit summary. Adding or editing any
detected notice therefore cannot pass as unreviewed drift.
The two named audit-control artifacts under `docs/memory/` and `policy/` are
excluded from notice discovery to avoid treating this evidence and the
self-referential policy file as third-party notices. The documentation is
independently content-pinned as required evidence; policy changes remain an
explicit reviewed diff.
The policy is evidence about repository state; it does not grant, revoke, or
change any copyright or license right.

`Cargo.lock` provides dependency identities, versions, sources, and checksums;
it does not contain authoritative dependency license terms. The policy therefore
keeps dependency license obligations explicitly unresolved and subject to manual
review rather than treating a successful inventory as legal clearance.

The report always sets `automatic_relicensing_ready=false`. A source-tree
scan cannot prove ownership of every contribution, contributor consent, or the
legal effect of historical license grants. Those questions require documented
rights evidence and, where appropriate, legal review.

No future change from the current repository license metadata should be
inferred from this audit alone. Third-party copyrights, notices, licenses and
attribution obligations must be preserved independently of the license chosen
for NNIS-owned code.
