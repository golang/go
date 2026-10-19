# Section 10: configuration schema (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 613-658 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 10: `bettergo.toml` CANONICAL CONFIGURATION SCHEMA:**

```toml
[project]
name = "order-engine"
edition = "2026"
strict_files = [".lgo", ".📜", ".⚡", ".ʕ◔ϖ◔ʔ"]

[build]
opt_level = "O2"                     # "O0" | "O1" | "O2" | "O3" | "Os" | "Oz"
gc = "greentea"                      # "greentea" | "zgc" | "immix" | "wasm" | "batch" | "epsilon"
synthetic_pgo = "standard"           # "off" | "standard" | "aggressive"
target_cpu = "deterministic"         # "deterministic" (v3/v8.2) | "native"
cgo = "restricted"                   # "forbid" | "restricted" | "allow"
cgo_allowlist = [
    "github.com/duckdb/duckdb-go/**",
]

[capabilities]
# Enforced across transitive third-party dependencies via cmd/link reachability audit
direct_reflect = "deny"
transitive_reflect = "warn"
reflect_allowlist = ["google.golang.org/protobuf/**"]
transitive_unsafe_allowlist = ["golang.org/x/sys/**", "github.com/klauspost/compress/**"]
transitive_exec_allowlist = []

[lints]
correctness = "forbid"               # Cannot be bypassed with //logos:allow
concurrency = "forbid"
slop = "deny"
complexity = "deny"
perf = "warn"
style = "warn"
pedantic = "allow"                   # Opt-in via --pedantic

[lints.thresholds]
max_nesting_depth = 3
max_cognitive_complexity = 15
max_function_lines = 60
max_function_params = 4
max_comptime_ast_nodes = 4096
max_ctfe_steps = 100000
```

---

