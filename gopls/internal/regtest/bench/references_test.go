// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import "testing"

func BenchmarkReferences(b *testing.B) {
	tests := []struct {
		repo   string
		file   string
		regexp string
	}{
		{"google-cloud-go", "httpreplay/httpreplay.go", `func (NewRecorder)`},
		{"istio", "pkg/config/model.go", "type (Meta)"},
		{"kubernetes", "pkg/controller/lookup_cache.go", "type (objectWithMeta)"},
		{"kuma", "pkg/events/interfaces.go", "type (Event)"},
		{"pkgsite", "internal/log/log.go", "func (Infof)"},
		{"starlark", "syntax/syntax.go", "type (Ident)"},
		{"tools", "internal/lsp/source/view.go", "type (Snapshot)"},
	}

	for _, test := range tests {
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			loc := env.RegexpSearch(test.file, test.regexp)
			env.Await(env.DoneWithOpen())
			env.References(loc) // pre-warm the query
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				env.References(loc)
			}
		})
	}
}
