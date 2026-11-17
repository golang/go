# Final installed tooling and runtime gates

Recorded on 2026-11-16 after the normal fresh make of e96a570da271c43a80e23bb1a188aac054acff25. The source tree is e237c399aae8f8065b724e8a0e27585686c7da18.

All six serial stages passed. No test skipped, no race report appeared, and the final tracked worktree remained clean. Before/after records are byte-identical and bind all 15 frozen source/test inputs, six installed tools, the retained cmd/go race binary, and pinned QEMU.

| Gate | Exit | Seconds | Observed scope |
|---|---:|---:|---|
| Installed overlay script | 0 | 5.636 | Frozen effective-source assertions and emitted program output overlay ok |
| Linker test build | 0 | 11.712 | Final installed compiler, exact final source |
| Linker components | 0 | 0.912 | 25 top-level tests / 75 cells; literal five-second child bound unchanged |
| Linker package vet | 0 | 1.914 | Normal final-tool package vet |
| Native capability scripts | 0 | 87.423 | chains, exec, policy, shortest_frames, test_scope |
| Sum cross-architecture script | 0 | 10.432 | Actual QEMU arm64 execution and identical 44-line native/arm64 output |

The capability transcript includes the actual final linker invocation with its -logos.cap manifest. Native cgo is enabled: the denied runtime.cgocall path and the allowed executable printing 42 both ran. The QEMU phase leaves CGO_ENABLED absent so the fixture derives the cross-target default. Its frozen script hash remains 7b5ffb83bceecf552577cf056132f9a86ca2518d86faa231f8e6d6f88adad167.

The retained cmd/go race binary was produced by the language lane from clean e96 source without an overlay. Its SHA-256 is 9645693e7edb1d9b4a1a02c0a74b158283db73158189b62c0c4feb45b3103f0a. The preflight records its original receipt and build command. A new linker test binary was built with the final installed compiler. Compiled binaries are intentionally excluded from the portable evidence.

Start with runs/installed-001/summary.json and transcript-review.json. Each stage retains inputs.json, result.json, and raw.log. The original plan.json is historical preparation; actual per-phase inputs record the corrected QEMU environment. static-binding.json proves the final workflow inputs are byte-identical to the recorded actionlint 1.7.12 / ShellCheck 0.11.0 GREEN and verifies both active successor manifests.

Verify this directory after copying with: sha256sum --check --strict checksums.sha256. The manifest uses paths relative to this directory. Exclude every .test binary. Do not rewrite original logs or preparation records.

These records establish the focused installed #336 and #441 gates and actual sum-crossarch execution. Full Go regression, hosted workflow acceptance, platform coverage, and stamped release parity remain separate requirements. No broader release-complete claim is made.
