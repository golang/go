// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements tests for various issues.

package types_test

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"internal/testenv"
	"sort"
	"strings"
	"testing"

	. "go/types"
)

func TestIssue5770(t *testing.T) {
	src := `package p; type S struct{T}`
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	conf := Config{Importer: importer.Default()}
	_, err = conf.Check(f.Name.Name, fset, []*ast.File{f}, nil) // do not crash
	want := "undeclared name: T"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("got: %v; want: %s", err, want)
	}
}

func TestIssue5849(t *testing.T) {
	src := `
package p
var (
	s uint
	_ = uint8(8)
	_ = uint16(16) << s
	_ = uint32(32 << s)
	_ = uint64(64 << s + s)
	_ = (interface{})("foo")
	_ = (interface{})(nil)
)`
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf Config
	types := make(map[ast.Expr]TypeAndValue)
	_, err = conf.Check(f.Name.Name, fset, []*ast.File{f}, &Info{Types: types})
	if err != nil {
		t.Fatal(err)
	}

	for x, tv := range types {
		var want Type
		switch x := x.(type) {
		case *ast.BasicLit:
			switch x.Value {
			case `8`:
				want = Typ[Uint8]
			case `16`:
				want = Typ[Uint16]
			case `32`:
				want = Typ[Uint32]
			case `64`:
				want = Typ[Uint] // because of "+ s", s is of type uint
			case `"foo"`:
				want = Typ[String]
			}
		case *ast.Ident:
			if x.Name == "nil" {
				want = Typ[UntypedNil]
			}
		}
		if want != nil && !Identical(tv.Type, want) {
			t.Errorf("got %s; want %s", tv.Type, want)
		}
	}
}

func TestIssue6413(t *testing.T) {
	src := `
package p
func f() int {
	defer f()
	go f()
	return 0
}
`
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf Config
	types := make(map[ast.Expr]TypeAndValue)
	_, err = conf.Check(f.Name.Name, fset, []*ast.File{f}, &Info{Types: types})
	if err != nil {
		t.Fatal(err)
	}

	want := Typ[Int]
	n := 0
	for x, tv := range types {
		if _, ok := x.(*ast.CallExpr); ok {
			if tv.Type != want {
				t.Errorf("%s: got %s; want %s", fset.Position(x.Pos()), tv.Type, want)
			}
			n++
		}
	}

	if n != 2 {
		t.Errorf("got %d CallExprs; want 2", n)
	}
}

func TestIssue7245(t *testing.T) {
	src := `
package p
func (T) m() (res bool) { return }
type T struct{} // receiver type after method declaration
`
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	var conf Config
	defs := make(map[*ast.Ident]Object)
	_, err = conf.Check(f.Name.Name, fset, []*ast.File{f}, &Info{Defs: defs})
	if err != nil {
		t.Fatal(err)
	}

	m := f.Decls[0].(*ast.FuncDecl)
	res1 := defs[m.Name].(*Func).Type().(*Signature).Results().At(0)
	res2 := defs[m.Type.Results.List[0].Names[0]].(*Var)

	if res1 != res2 {
		t.Errorf("got %s (%p) != %s (%p)", res1, res2, res1, res2)
	}
}

// This tests that uses of existing vars on the LHS of an assignment
// are Uses, not Defs; and also that the (illegal) use of a non-var on
// the LHS of an assignment is a Use nonetheless.
func TestIssue7827(t *testing.T) {
	const src = `
package p
func _() {
	const w = 1        // defs w
        x, y := 2, 3       // defs x, y
        w, x, z := 4, 5, 6 // uses w, x, defs z; error: cannot assign to w
        _, _, _ = x, y, z  // uses x, y, z
}
`
	const want = `L3 defs func p._()
L4 defs const w untyped int
L5 defs var x int
L5 defs var y int
L6 defs var z int
L6 uses const w untyped int
L6 uses var x int
L7 uses var x int
L7 uses var y int
L7 uses var z int`

	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	// don't abort at the first error
	conf := Config{Error: func(err error) { t.Log(err) }}
	defs := make(map[*ast.Ident]Object)
	uses := make(map[*ast.Ident]Object)
	_, err = conf.Check(f.Name.Name, fset, []*ast.File{f}, &Info{Defs: defs, Uses: uses})
	if s := fmt.Sprint(err); !strings.HasSuffix(s, "cannot assign to w") {
		t.Errorf("Check: unexpected error: %s", s)
	}

	var facts []string
	for id, obj := range defs {
		if obj != nil {
			fact := fmt.Sprintf("L%d defs %s", fset.Position(id.Pos()).Line, obj)
			facts = append(facts, fact)
		}
	}
	for id, obj := range uses {
		fact := fmt.Sprintf("L%d uses %s", fset.Position(id.Pos()).Line, obj)
		facts = append(facts, fact)
	}
	sort.Strings(facts)

	got := strings.Join(facts, "\n")
	if got != want {
		t.Errorf("Unexpected defs/uses\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// This tests that the package associated with the types.Object.Pkg method
// is the type's package independent of the order in which the imports are
// listed in the sources src1, src2 below.
// The actual issue is in go/internal/gcimporter which has a corresponding
// test; we leave this test here to verify correct behavior at the go/types
// level.
func TestIssue13898(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const src0 = `
package main

import "go/types"

func main() {
	var info types.Info
	for _, obj := range info.Uses {
		_ = obj.Pkg()
	}
}
`
	// like src0, but also imports go/importer
	const src1 = `
package main

import (
	"go/types"
	_ "go/importer"
)

func main() {
	var info types.Info
	for _, obj := range info.Uses {
		_ = obj.Pkg()
	}
}
`
	// like src1 but with different import order
	// (used to fail with this issue)
	const src2 = `
package main

import (
	_ "go/importer"
	"go/types"
)

func main() {
	var info types.Info
	for _, obj := range info.Uses {
		_ = obj.Pkg()
	}
}
`
	f := func(test, src string) {
		f, err := parser.ParseFile(fset, "", src, 0)
		if err != nil {
			t.Fatal(err)
		}
		cfg := Config{Importer: importer.Default()}
		info := Info{Uses: make(map[*ast.Ident]Object)}
		_, err = cfg.Check("main", fset, []*ast.File{f}, &info)
		if err != nil {
			t.Fatal(err)
		}

		var pkg *Package
		count := 0
		for id, obj := range info.Uses {
			if id.Name == "Pkg" {
				pkg = obj.Pkg()
				count++
			}
		}
		if count != 1 {
			t.Fatalf("%s: got %d entries named Pkg; want 1", test, count)
		}
		if pkg.Name() != "types" {
			t.Fatalf("%s: got %v; want package types", test, pkg)
		}
	}

	f("src0", src0)
	f("src1", src1)
	f("src2", src2)
}
