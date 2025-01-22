// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"go/token"
	"path/filepath"
	"runtime"
	"testing"

	. "go/types"
)

// BenchmarkLookupFieldOrMethod measures types.LookupFieldOrMethod performance.
// LookupFieldOrMethod is a performance hotspot for both type-checking and
// external API calls.
func BenchmarkLookupFieldOrMethod(b *testing.B) {
	// Choose an arbitrary, large package.
	path := filepath.Join(runtime.GOROOT(), "src", "net", "http")

	fset := token.NewFileSet()
	files, err := pkgFiles(fset, path)
	if err != nil {
		b.Fatal(err)
	}

	conf := Config{
		Importer: defaultImporter(fset),
	}

	pkg, err := conf.Check("http", fset, files, nil)
	if err != nil {
		b.Fatal(err)
	}

	scope := pkg.Scope()
	names := scope.Names()

	// Look up an arbitrary name for each type referenced in the package scope.
	lookup := func() {
		for _, name := range names {
			typ := scope.Lookup(name).Type()
			LookupFieldOrMethod(typ, true, pkg, "m")
		}
	}

	// Perform a lookup once, to ensure that any lazily-evaluated state is
	// complete.
	lookup()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lookup()
	}
}
