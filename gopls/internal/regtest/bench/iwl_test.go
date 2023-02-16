// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// BenchmarkInitialWorkspaceLoad benchmarks the initial workspace load time for
// a new editing session.
func BenchmarkInitialWorkspaceLoad(b *testing.B) {
	repo := repos["tools"]
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// Exclude the time to set up the env from the benchmark time, as this may
		// involve installing gopls and/or checking out the repo dir.
		b.StopTimer()
		env := repo.newEnv(b)
		b.StartTimer()

		env.OpenFile("internal/lsp/diagnostics.go")
		env.Await(InitialWorkspaceLoad)
		b.StopTimer()
		params := &protocol.ExecuteCommandParams{
			Command: command.MemStats.ID(),
		}
		var memstats command.MemStatsResult
		env.ExecuteCommand(params, &memstats)
		b.ReportMetric(float64(memstats.HeapAlloc), "alloc_bytes")
		b.ReportMetric(float64(memstats.HeapInUse), "in_use_bytes")
		env.Close()
		b.StartTimer()
	}
}
