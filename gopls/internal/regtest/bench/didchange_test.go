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
//
// Uses -workdir and -file to control where the edits occur.
func BenchmarkDidChange(b *testing.B) {
	env := benchmarkEnv(b)
	env.OpenFile(*file)
	env.Await(env.DoneWithOpen())

	// Insert the text we'll be modifying at the top of the file.
	env.EditBuffer(*file, protocol.TextEdit{NewText: "// __REGTEST_PLACEHOLDER_0__\n"})

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		env.EditBuffer(*file, protocol.TextEdit{
			Range: protocol.Range{
				Start: protocol.Position{Line: 0, Character: 0},
				End:   protocol.Position{Line: 1, Character: 0},
			},
			// Increment the placeholder text, to ensure cache misses.
			NewText: fmt.Sprintf("// __REGTEST_PLACEHOLDER_%d__\n", i+1),
		})
		env.Await(env.StartedChange())
	}
}
