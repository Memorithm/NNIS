#!/usr/bin/env python3
"""P0 physical qualification bundle v3 using the TinyLlama F16-oracle campaign v2."""

from __future__ import annotations

from pathlib import Path

import run_p0_physical_qualification_bundle as base

_original_run_stream = base.run_stream


def run_stream_v3(command: list[str], *, cwd: Path) -> None:
    rewritten = list(command)
    if len(rewritten) >= 2 and rewritten[1].endswith("run_tinyllama_massive_campaign.py"):
        rewritten[1] = str(Path(rewritten[1]).with_name("run_tinyllama_massive_campaign_v2.py"))
    _original_run_stream(rewritten, cwd=cwd)


def main() -> None:
    base.BUNDLE_KIND = "nnis-p0-physical-qualification-bundle-v3"
    base.SCHEMA_VERSION = 3
    base.run_stream = run_stream_v3
    base.main()


if __name__ == "__main__":
    main()
