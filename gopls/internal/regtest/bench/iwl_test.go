// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"testing"

	"golang.org/x/tools/gopls/internal/lsp/command"
	"golang.org/x/tools/gopls/internal/lsp/fake"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// BenchmarkInitialWorkspaceLoad benchmarks the initial workspace load time for
// a new editing session.
func BenchmarkInitialWorkspaceLoad(b *testing.B) {
	tests := []struct {
		repo string
		file string
	}{
		{"google-cloud-go", "httpreplay/httpreplay.go"},
		{"istio", "pkg/fuzz/util.go"},
		{"kubernetes", "pkg/controller/lookup_cache.go"},
		{"kuma", "api/generic/insights.go"},
		{"oracle", "dataintegration/data_type.go"},
		{"pkgsite", "internal/frontend/server.go"},
		{"starlark", "starlark/eval.go"},
		{"tools", "internal/lsp/cache/snapshot.go"},
		{"hashiform", "internal/provider/provider.go"},
	}

	for _, test := range tests {
		b.Run(test.repo, func(b *testing.B) {
			repo := getRepo(b, test.repo)
			// get the (initialized) shared env to ensure the cache is warm.
			// Reuse its GOPATH so that we get cache hits for things in the module
			// cache.
			sharedEnv := repo.sharedEnv(b)
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				doIWL(b, sharedEnv.Sandbox.GOPATH(), repo, test.file)
			}
		})
	}
}

func doIWL(b *testing.B, gopath string, repo *repo, file string) {
	// Exclude the time to set up the env from the benchmark time, as this may
	// involve installing gopls and/or checking out the repo dir.
	b.StopTimer()
	config := fake.EditorConfig{Env: map[string]string{"GOPATH": gopath}}
	env := repo.newEnv(b, config, "iwl", true)
	defer env.Close()
	b.StartTimer()

	// Note: in the future, we may need to open a file in order to cause gopls to
	// start loading the workspace.

	env.Await(InitialWorkspaceLoad)

	if env.Editor.HasCommand(command.MemStats.ID()) {
		b.StopTimer()
		params := &protocol.ExecuteCommandParams{
			Command: command.MemStats.ID(),
		}
		var memstats command.MemStatsResult
		env.ExecuteCommand(params, &memstats)
		b.ReportMetric(float64(memstats.HeapAlloc), "alloc_bytes")
		b.ReportMetric(float64(memstats.HeapInUse), "in_use_bytes")
		b.ReportMetric(float64(memstats.TotalAlloc), "total_alloc_bytes")
		b.StartTimer()
	}
}
