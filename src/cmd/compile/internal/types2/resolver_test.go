// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"internal/testenv"
	"sort"
	"testing"

	. "cmd/compile/internal/types2"
)

type resolveTestImporter struct {
	importer ImporterFrom
	imported map[string]bool
}

func (imp *resolveTestImporter) Import(string) (*Package, error) {
	panic("should not be called")
}

func (imp *resolveTestImporter) ImportFrom(path, srcDir string, mode ImportMode) (*Package, error) {
	if mode != 0 {
		panic("mode must be 0")
	}
	if imp.importer == nil {
		imp.importer = defaultImporter().(ImporterFrom)
		imp.imported = make(map[string]bool)
	}
	pkg, err := imp.importer.ImportFrom(path, srcDir, mode)
	if err != nil {
		return nil, err
	}
	imp.imported[path] = true
	return pkg, nil
}

func TestResolveIdents(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	sources := []string{
		`
		package p
		import "fmt"
		import "math"
		const pi = math.Pi
		func sin(x float64) float64 {
			return math.Sin(x)
		}
		var Println = fmt.Println
		`,
		`
		package p
		import "fmt"
		type errorStringer struct { fmt.Stringer; error }
		func f() string {
			_ = "foo"
			return fmt.Sprintf("%d", g())
		}
		func g() (x int) { return }
		`,
		`
		package p
		import . "go/parser"
		import "sync"
		func h() Mode { return ImportsOnly }
		var _, x int = 1, 2
		func init() {}
		type T struct{ *sync.Mutex; a, b, c int}
		type I interface{ m() }
		var _ = T{a: 1, b: 2, c: 3}
		func (_ T) m() {}
		func (T) _() {}
		var i I
		var _ = i.m
		func _(s []int) { for i, x := range s { _, _ = i, x } }
		func _(x interface{}) {
			switch x := x.(type) {
			case int:
				_ = x
			}
			switch {} // implicit 'true' tag
		}
		`,
		`
		package p
		type S struct{}
		func (T) _() {}
		func (T) _() {}
		`,
		`
		package p
		func _() {
		L0:
		L1:
			goto L0
			for {
				goto L1
			}
			if true {
				goto L2
			}
		L2:
		}
		`,
	}

	pkgnames := []string{
		"fmt",
		"math",
	}

	// parse package files
	var files []*syntax.File
	for _, src := range sources {
		files = append(files, mustParse(src))
	}

	// resolve and type-check package AST
	importer := new(resolveTestImporter)
	conf := Config{Importer: importer}
	uses := make(map[*syntax.Name]Object)
	defs := make(map[*syntax.Name]Object)
	_, err := conf.Check("testResolveIdents", files, &Info{Defs: defs, Uses: uses})
	if err != nil {
		t.Fatal(err)
	}

	// check that all packages were imported
	for _, name := range pkgnames {
		if !importer.imported[name] {
			t.Errorf("package %s not imported", name)
		}
	}

	// check that qualified identifiers are resolved
	for _, f := range files {
		syntax.Inspect(f, func { n ->
			if s, ok := n.(*syntax.SelectorExpr); ok {
				if x, ok := s.X.(*syntax.Name); ok {
					obj := uses[x]
					if obj == nil {
						t.Errorf("%s: unresolved qualified identifier %s", x.Pos(), x.Value)
						return false
					}
					if _, ok := obj.(*PkgName); ok && uses[s.Sel] == nil {
						t.Errorf("%s: unresolved selector %s", s.Sel.Pos(), s.Sel.Value)
						return false
					}
					return false
				}
				return true
			}
			return true
		})
	}

	for id, obj := range uses {
		if obj == nil {
			t.Errorf("%s: Uses[%s] == nil", id.Pos(), id.Value)
		}
	}

	// Check that each identifier in the source is found in uses or defs or both.
	// We need the foundUses/Defs maps (rather than just deleting the found objects
	// from the uses and defs maps) because syntax.Walk traverses shared nodes multiple
	// times (e.g. types in field lists such as "a, b, c int").
	foundUses := make(map[*syntax.Name]bool)
	foundDefs := make(map[*syntax.Name]bool)
	var both []string
	for _, f := range files {
		syntax.Inspect(f, func { n ->
			if x, ok := n.(*syntax.Name); ok {
				var objects int
				if _, found := uses[x]; found {
					objects |= 1
					foundUses[x] = true
				}
				if _, found := defs[x]; found {
					objects |= 2
					foundDefs[x] = true
				}
				switch objects {
				case 0:
					t.Errorf("%s: unresolved identifier %s", x.Pos(), x.Value)
				case 3:
					both = append(both, x.Value)
				}
				return false
			}
			return true
		})
	}

	// check the expected set of idents that are simultaneously uses and defs
	sort.Strings(both)
	if got, want := fmt.Sprint(both), "[Mutex Stringer error]"; got != want {
		t.Errorf("simultaneous uses/defs = %s, want %s", got, want)
	}

	// any left-over identifiers didn't exist in the source
	for x := range uses {
		if !foundUses[x] {
			t.Errorf("%s: identifier %s not present in source", x.Pos(), x.Value)
		}
	}
	for x := range defs {
		if !foundDefs[x] {
			t.Errorf("%s: identifier %s not present in source", x.Pos(), x.Value)
		}
	}

	// TODO(gri) add tests to check ImplicitObj callbacks
}
