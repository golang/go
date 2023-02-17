// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"fmt"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// BenchmarkDidChange benchmarks modifications of a single file by making
// synthetic modifications in a comment. It controls pacing by waiting for the
// server to actually start processing the didChange notification before
// proceeding. Notably it does not wait for diagnostics to complete.
func BenchmarkDidChange(b *testing.B) {
	tests := []struct {
		repo string
		file string
	}{
		{"istio", "pkg/fuzz/util.go"},
		{"kubernetes", "pkg/controller/lookup_cache.go"},
		{"kuma", "api/generic/insights.go"},
		{"pkgsite", "internal/frontend/server.go"},
		{"starlark", "starlark/eval.go"},
		{"tools", "internal/lsp/cache/snapshot.go"},
	}

	for _, test := range tests {
		edits := 0 // bench function may execute multiple times
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			env.AfterChange()
			// Insert the text we'll be modifying at the top of the file.
			env.EditBuffer(test.file, protocol.TextEdit{NewText: "// __REGTEST_PLACEHOLDER_0__\n"})
			env.AfterChange()
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				edits++
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
