// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"testing"
)

func BenchmarkDefinition(b *testing.B) {
	tests := []struct {
		repo   string
		file   string
		regexp string
	}{
		{"istio", "pkg/config/model.go", `gogotypes\.(MarshalAny)`},
		{"google-cloud-go", "httpreplay/httpreplay.go", `proxy\.(ForRecording)`},
		{"kubernetes", "pkg/controller/lookup_cache.go", `hashutil\.(DeepHashObject)`},
		{"kuma", "api/generic/insights.go", `proto\.(Message)`},
		{"pkgsite", "internal/log/log.go", `derrors\.(Wrap)`},
		{"starlark", "starlark/eval.go", "prog.compiled.(Encode)"},
		{"tools", "internal/lsp/cache/check.go", `(snapshot)\) buildKey`},
	}

	for _, test := range tests {
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			loc := env.RegexpSearch(test.file, test.regexp)
			env.Await(env.DoneWithOpen())
			env.GoToDefinition(loc) // pre-warm the query, and open the target file
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				env.GoToDefinition(loc) // pre-warm the query
			}
		})
	}
}
