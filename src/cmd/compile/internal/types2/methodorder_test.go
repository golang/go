// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/testimporter"
	. "cmd/compile/internal/types2"
)

// TestImportedMethodOrder checks that methods of types imported from
// export data appear in source order, as required by objectpath
// (go.dev/issue/81188).
func TestImportedMethodOrder(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const src = `package methodorder

// T mixes generic and non-generic methods.
type T struct{}

func (T) A()               {}
func (T) B[X any]()        {}
func (T) C()               {}
func (T) D[X, Y any](X, Y) {}
func (T) E()               {}

// U's underlying type refers to V, whose method body calls a generic
// method of U, so encoding U's underlying type encodes U.G before the
// rest of U.
type U struct{ v *V }

type V struct{}

func (v *V) M(u *U) int { return u.G[int]() }

func (*U) A()            {}
func (*U) B()            {}
func (*U) G[X any]() int { return 0 }
func (*U) Z()            {}
`
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module methodorder\n\ngo 1.27\n"), 0666); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "methodorder.go"), []byte(src), 0666); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir) // so that the importer's go build uses dir's module

	pkg, err := testimporter.NewImporter().ImportFrom(".", dir, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		typ  string
		want string
	}{
		{"T", "A B C D E"},
		{"U", "A B G Z"},
	} {
		named := pkg.Scope().Lookup(test.typ).Type().(*Named)
		var names []string
		for i := range named.NumMethods() {
			names = append(names, named.Method(i).Name())
		}
		if got := strings.Join(names, " "); got != test.want {
			t.Errorf("methods of %s = %s; want %s", test.typ, got, test.want)
		}
	}
}
