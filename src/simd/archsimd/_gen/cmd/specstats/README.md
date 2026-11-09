QUALITY: AI-generated with minimal human review. This is an internally-facing
tool and will be retired after the spec transition.

This tool prints the progress metrics report for the transition from
disjoint generators to `internal/spec` as described in SPEC-TRANSITION.md:
`go run ./cmd/specstats`. The report is not checked in; `go test
./cmd/specstats` enforces the counts that must stay at zero.
