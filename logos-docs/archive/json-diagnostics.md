# Section 16: JSON diagnostics schema (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 928-960 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 16: `--json-diagnostics` SCHEMA FOR AI AGENTS & SUPERVISORS (`betterlint/json.go`):**

When invoked with `logos lint --json-diagnostics ./...` or `logos build --json-diagnostics`, all compiler and linter errors emit deterministic single-line JSON records for zero-shot LLM self-healing:

```json
{
  "schema_version": "1.0",
  "severity": "forbid",
  "rule_id": "correctness::nil_deref_unproven",
  "file": "pkg/auth/session.lgo",
  "span": { "start_line": 42, "start_col": 14, "end_line": 42, "end_col": 26, "byte_offset": 1184, "byte_len": 12 },
  "message": "cannot dereference nullable type '*User?' without proving '!= nil'",
  "provenance": {
    "origin_expr": "m[sessionID]",
    "origin_line": 40,
    "inferred_type": "*User? (interface { *User | nil })"
  },
  "autofix_candidates": [
    {
      "description": "Use optional chaining with default fallback",
      "replacement": "user?.Name ?? \"anonymous\""
    },
    {
      "description": "Insert early-return nil guard",
      "insert_before_line": 42,
      "replacement": "if user == nil {\n\treturn \"\", ErrSessionNotFound\n}\n"
    }
  ]
}
```

---

