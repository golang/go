// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
)

// BenchmarkTyping simulates typing steadily in a single file at different
// paces.
//
// The key metric for this benchmark is not latency, but cpu_seconds per
// operation.
func BenchmarkTyping(b *testing.B) {
	for _, test := range didChangeTests {
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			defer closeBuffer(b, env, test.file)

			// Insert the text we'll be modifying at the top of the file.
			env.EditBuffer(test.file, protocol.TextEdit{NewText: "// __REGTEST_PLACEHOLDER_0__\n"})
			env.AfterChange()

			delays := []time.Duration{
				10 * time.Millisecond,  // automated changes
				50 * time.Millisecond,  // very fast mashing, or fast key sequences
				150 * time.Millisecond, // avg interval for 80wpm typing.
			}

			for _, delay := range delays {
				b.Run(delay.String(), func(b *testing.B) {
					if stopAndRecord := startProfileIfSupported(b, env, qualifiedName(test.repo, "typing")); stopAndRecord != nil {
						defer stopAndRecord()
					}
					ticker := time.NewTicker(delay)
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
						<-ticker.C
					}
					b.StopTimer()
					ticker.Stop()
					env.AfterChange() // wait for all change processing to complete
				})
			}
		})
	}
}
