// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// Use a global edit counter as bench function may execute multiple times, and
// we want to avoid cache hits. Use time.Now to also avoid cache hits from the
// shared file cache.
var editID int64 = time.Now().UnixNano()

type changeTest struct {
	repo string
	file string
}

var didChangeTests = []changeTest{
	{"google-cloud-go", "internal/annotate.go"},
	{"istio", "pkg/fuzz/util.go"},
	{"kubernetes", "pkg/controller/lookup_cache.go"},
	{"kuma", "api/generic/insights.go"},
	{"pkgsite", "internal/frontend/server.go"},
	{"starlark", "starlark/eval.go"},
	{"tools", "internal/lsp/cache/snapshot.go"},
}

// BenchmarkDidChange benchmarks modifications of a single file by making
// synthetic modifications in a comment. It controls pacing by waiting for the
// server to actually start processing the didChange notification before
// proceeding. Notably it does not wait for diagnostics to complete.
func BenchmarkDidChange(b *testing.B) {
	for _, test := range didChangeTests {
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			// Insert the text we'll be modifying at the top of the file.
			env.EditBuffer(test.file, protocol.TextEdit{NewText: "// __REGTEST_PLACEHOLDER_0__\n"})
			env.AfterChange()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				edits := atomic.AddInt64(&editID, 1)
				env.EditBuffer(test.file, protocol.TextEdit{
					Range: protocol.Range{
						Start: protocol.Position{Line: 0, Character: 0},
						End:   protocol.Position{Line: 1, Character: 0},
					},
					// Increment the placeholder text, to ensure cache misses.
					NewText: fmt.Sprintf("// __REGTEST_PLACEHOLDER_%d__\n", edits),
				})
				env.Await(env.StartedChange())
			}
		})
	}
}

func BenchmarkDiagnoseChange(b *testing.B) {
	for _, test := range didChangeTests {
		runChangeDiagnosticsBenchmark(b, test, false)
	}
}

// TODO(rfindley): add a benchmark for with a metadata-affecting change, when
// this matters.
func BenchmarkDiagnoseSave(b *testing.B) {
	for _, test := range didChangeTests {
		runChangeDiagnosticsBenchmark(b, test, true)
	}
}

// runChangeDiagnosticsBenchmark runs a benchmark to edit the test file and
// await the resulting diagnostics pass. If save is set, the file is also saved.
func runChangeDiagnosticsBenchmark(b *testing.B, test changeTest, save bool) {
	b.Run(test.repo, func(b *testing.B) {
		sharedEnv := getRepo(b, test.repo).sharedEnv(b)
		config := fake.EditorConfig{
			Env: map[string]string{
				"GOPATH": sharedEnv.Sandbox.GOPATH(),
			},
			Settings: map[string]interface{}{
				"diagnosticsDelay": "0s",
			},
		}
		// Use a new env to avoid the diagnostic delay: we want to measure how
		// long it takes to produce the diagnostics.
		env := getRepo(b, test.repo).newEnv(b, "diagnoseSave", config)
		defer env.Close()
		env.OpenFile(test.file)
		// Insert the text we'll be modifying at the top of the file.
		env.EditBuffer(test.file, protocol.TextEdit{NewText: "// __REGTEST_PLACEHOLDER_0__\n"})
		if save {
			env.SaveBuffer(test.file)
		}
		env.AfterChange()
		b.ResetTimer()

		// We must use an extra subtest layer here, so that we only set up the
		// shared env once (otherwise we pay additional overhead and the profiling
		// flags don't work).
		b.Run("diagnose", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				edits := atomic.AddInt64(&editID, 1)
				env.EditBuffer(test.file, protocol.TextEdit{
					Range: protocol.Range{
						Start: protocol.Position{Line: 0, Character: 0},
						End:   protocol.Position{Line: 1, Character: 0},
					},
					// Increment the placeholder text, to ensure cache misses.
					NewText: fmt.Sprintf("// __REGTEST_PLACEHOLDER_%d__\n", edits),
				})
				if save {
					env.SaveBuffer(test.file)
				}
				env.AfterChange()
			}
		})
	})
}
