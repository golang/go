// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package bench

import (
	"testing"

	. "golang.org/x/tools/gopls/internal/lsp/regtest"
)

// BenchmarkReload benchmarks reloading a file metadata after a change to an import.
//
// This ensures we are able to diagnose a changed file without reloading all
// invalidated packages. See also golang/go#61344
func BenchmarkReload(b *testing.B) {
	// TODO(rfindley): add more tests, make this test table-driven
	const (
		repo = "kubernetes"
		// pkg/util/hash is transitively imported by a large number of packages.
		// We should not need to reload those packages to get a diagnostic.
		file = "pkg/util/hash/hash.go"
	)
	b.Run(repo, func(b *testing.B) {
		env := getRepo(b, repo).sharedEnv(b)

		env.OpenFile(file)
		defer closeBuffer(b, env, file)

		env.AfterChange()

		if stopAndRecord := startProfileIfSupported(b, env, qualifiedName(repo, "reload")); stopAndRecord != nil {
			defer stopAndRecord()
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Change the "hash" import. This may result in cache hits, but that's
			// OK: the goal is to ensure that we don't reload more than just the
			// current package.
			env.RegexpReplace(file, `"hash"`, `"hashx"`)
			// Note: don't use env.AfterChange() here: we only want to await the
			// first diagnostic.
			//
			// Awaiting a full diagnosis would await diagnosing everything, which
			// would require reloading everything.
			env.Await(Diagnostics(ForFile(file)))
			env.RegexpReplace(file, `"hashx"`, `"hash"`)
			env.Await(NoDiagnostics(ForFile(file)))
		}
	})
}
