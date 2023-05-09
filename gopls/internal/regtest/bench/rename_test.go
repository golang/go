// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"fmt"
	"testing"
)

func BenchmarkRename(b *testing.B) {
	tests := []struct {
		repo     string
		file     string
		regexp   string
		baseName string
	}{
		{"google-cloud-go", "httpreplay/httpreplay.go", `func (NewRecorder)`, "NewRecorder"},
		{"istio", "pkg/config/model.go", `(Namespace) string`, "Namespace"},
		{"kubernetes", "pkg/controller/lookup_cache.go", `hashutil\.(DeepHashObject)`, "DeepHashObject"},
		{"kuma", "pkg/events/interfaces.go", `Delete`, "Delete"},
		{"pkgsite", "internal/log/log.go", `func (Infof)`, "Infof"},
		{"starlark", "starlark/eval.go", `Program\) (Filename)`, "Filename"},
		{"tools", "internal/lsp/cache/snapshot.go", `meta \*(metadataGraph)`, "metadataGraph"},
	}

	for _, test := range tests {
		names := 0 // bench function may execute multiple times
		b.Run(test.repo, func(b *testing.B) {
			env := getRepo(b, test.repo).sharedEnv(b)
			env.OpenFile(test.file)
			loc := env.RegexpSearch(test.file, test.regexp)
			env.Await(env.DoneWithOpen())
			env.Rename(loc, test.baseName+"X") // pre-warm the query
			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				names++
				newName := fmt.Sprintf("%s%d", test.baseName, names)
				env.Rename(loc, newName)
			}
		})
	}
}
