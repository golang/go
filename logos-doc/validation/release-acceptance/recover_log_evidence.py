#!/usr/bin/env python3
"""Recover one verified evidence archive from a downloaded hosted job log."""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
from pathlib import Path


def recover(log: Path, job: str) -> tuple[bytes, str]:
    if re.fullmatch(r"[A-Za-z0-9_-]+", job) is None:
        raise ValueError("job must be a workflow job identifier")
    start = f"LOGOS_RELEASE_EVIDENCE_BEGIN {job}"
    end = f"LOGOS_RELEASE_EVIDENCE_END {job}"
    active = False
    ended = False
    expected = ""
    chunks: list[str] = []
    for line in log.read_text(errors="strict").splitlines():
        if line.endswith(start):
            if active or ended:
                raise ValueError("ambiguous duplicate evidence bundle")
            active = True
            continue
        if line.endswith(end):
            if not active:
                raise ValueError("evidence end without a matching start")
            active = False
            ended = True
            continue
        if not active:
            continue
        if not expected:
            match = re.search(r"(?:^|\s)([a-f0-9]{64})  \S+/evidence\.tar\.gz$", line)
            if match is None:
                raise ValueError("missing evidence archive digest")
            expected = match[1]
            continue
        payload = line.rsplit(" ", 1)[-1]
        if re.fullmatch(r"[A-Za-z0-9+/=]{1,76}", payload) is None:
            raise ValueError("unexpected data inside evidence bundle")
        chunks.append(payload)
    if active or not ended or not expected or not chunks:
        raise ValueError("incomplete evidence bundle")
    data = base64.b64decode("".join(chunks), validate=True)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("archive digest does not match retained evidence")
    return data, expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_log", type=Path)
    parser.add_argument("job")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data, digest = recover(args.job_log, args.job)
    with args.output.open("xb") as destination:
        destination.write(data)
    print(f"{digest}  {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
