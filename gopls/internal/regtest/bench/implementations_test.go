// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import "testing"

func BenchmarkImplementations(b *testing.B) {
	tests := []struct {
		repo   string
		file   string
		regexp string
	}{
		{"google-cloud-go", "httpreplay/httpreplay.go", `type (Recorder)`},
		{"istio", "pkg/config/mesh/watcher.go", `type (Watcher)`},
		{"kubernetes", "pkg/controller/lookup_cache.go", `objectWithMeta`},
		{"kuma", "api/generic/insights.go", `type (Insight)`},
		{"pkgsite", "internal/datasource.go", `type (DataSource)`},
		{"starlark", "syntax/syntax.go", `type (Expr)`},
		{"tools", "internal/lsp/source/view.go", `type (Snapshot)`},
	}

	for _, test := range tests {
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			loc := env.RegexpSearch(test.file, test.regexp)
			env.Await(env.DoneWithOpen())
			env.Implementations(loc) // pre-warm the query
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				env.Implementations(loc)
			}
		})
	}
}
