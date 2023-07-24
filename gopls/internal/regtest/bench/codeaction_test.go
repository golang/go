// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"fmt"
	"sync/atomic"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

func BenchmarkCodeAction(b *testing.B) {
	for _, test := range didChangeTests {
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			defer closeBuffer(b, env, test.file)
			env.AfterChange()

			env.CodeAction(test.file, nil) // pre-warm

			b.ResetTimer()

			if stopAndRecord := startProfileIfSupported(b, env, qualifiedName(test.repo, "hover")); stopAndRecord != nil {
				defer stopAndRecord()
			}

			for i := 0; i < b.N; i++ {
				env.CodeAction(test.file, nil)
			}
		})
	}
}

func BenchmarkCodeActionFollowingEdit(b *testing.B) {
	for _, test := range didChangeTests {
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			defer closeBuffer(b, env, test.file)
			env.EditBuffer(test.file, protocol.TextEdit{NewText: "// __REGTEST_PLACEHOLDER_0__\n"})
			env.AfterChange()

			env.CodeAction(test.file, nil) // pre-warm

			b.ResetTimer()

			if stopAndRecord := startProfileIfSupported(b, env, qualifiedName(test.repo, "hover")); stopAndRecord != nil {
				defer stopAndRecord()
			}

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
				env.CodeAction(test.file, nil)
			}
		})
	}
}
