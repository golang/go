// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	. "cmd/compile/internal/types2"
)

// BenchmarkLookupFieldOrMethod measures types.LookupFieldOrMethod performance.
// LookupFieldOrMethod is a performance hotspot for both type-checking and
// external API calls.
func BenchmarkLookupFieldOrMethod(b *testing.B) {
	// Choose an arbitrary, large package.
	path := filepath.Join(runtime.GOROOT(), "src", "net", "http")

	files, err := pkgFiles(path)
	if err != nil {
		b.Fatal(err)
	}

	conf := Config{
		Importer: defaultImporter(),
	}

	pkg, err := conf.Check("http", files, nil)
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
	for b.Loop() {
		lookup()
	}
}

// TestLookupManyMethods tests method lookup on types with enough methods
// to be looked up by name, concurrently and after adding a method.
func TestLookupManyMethods(t *testing.T) {
	const n = 200

	// The package declares an interface I with methods M0 to M199, a struct
	// type T and a generic type G that each implement them, and a struct S
	// that gets them by embedding T. The two var declarations check that S
	// and G[int] implement I.
	var src strings.Builder
	src.WriteString("package p; type I interface { ")
	for i := range n {
		fmt.Fprintf(&src, "M%d(); ", i)
	}
	src.WriteString("}; type T struct{}; type G[P any] struct{}; ")
	for i := range n {
		fmt.Fprintf(&src, "func (T) M%d() {}; func (G[P]) M%d() {}; ", i, i)
	}
	src.WriteString("type S struct{ T }; var _ I = S{}; var _ I = G[int]{}")
	pkg := mustTypecheck(src.String(), nil, nil)

	// S finds its methods through the embedded T (index length 2); G[string]
	// is an instance whose methods are created lazily.
	iface := pkg.Scope().Lookup("I").Type().Underlying().(*Interface)
	S := pkg.Scope().Lookup("S").Type()
	G, err := Instantiate(nil, pkg.Scope().Lookup("G").Type(), []Type{Typ[String]}, true)
	if err != nil {
		t.Fatal(err)
	}

	// Two goroutines per type look up every method by name, a missing name,
	// and the interface, so the method index is built under contention. The
	// method found for Mi must be the i-th one.
	var wg sync.WaitGroup
	for _, typ := range []Type{S, S, G, G} {
		wg.Go(func() {
			for i := range n {
				name := fmt.Sprintf("M%d", i)
				obj, index, _ := LookupFieldOrMethod(typ, false, pkg, name)
				if obj == nil || obj.Name() != name || index[len(index)-1] != i {
					t.Errorf("LookupFieldOrMethod(%s, %s) = %v, %v", typ, name, obj, index)
				}
			}
			if obj, _, _ := LookupFieldOrMethod(typ, false, pkg, "M"); obj != nil {
				t.Errorf("LookupFieldOrMethod(%s, M) = %v, want nil", typ, obj)
			}
			if !Implements(typ, iface) {
				t.Errorf("%s does not implement I", typ)
			}
		})
	}
	wg.Wait()

	// A method added to T after the index was built must be found too.
	T := pkg.Scope().Lookup("T").Type().(*Named)
	m := NewFunc(nopos, pkg, "M", NewSignatureType(NewVar(nopos, pkg, "", T), nil, nil, nil, nil, false))
	T.AddMethod(m)
	if obj, _, _ := LookupFieldOrMethod(T, false, pkg, "M"); obj != m {
		t.Errorf("LookupFieldOrMethod(T, M) = %v after AddMethod, want %v", obj, m)
	}
}
