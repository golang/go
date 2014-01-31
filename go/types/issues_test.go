// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements tests for various issues.

package types_test

import (
	"go/ast"
	"go/parser"
	"strings"
	"testing"

	_ "code.google.com/p/go.tools/go/gcimporter"
	. "code.google.com/p/go.tools/go/types"
)

func TestIssue5770(t *testing.T) {
	src := `package p; type S struct{T}`
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Check(f.Name.Name, fset, []*ast.File{f}) // do not crash
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
	objects := make(map[*ast.Ident]Object)
	_, err = conf.Check(f.Name.Name, fset, []*ast.File{f}, &Info{Objects: objects})
	if err != nil {
		t.Fatal(err)
	}

	m := f.Decls[0].(*ast.FuncDecl)
	res1 := objects[m.Name].(*Func).Type().(*Signature).Results().At(0)
	res2 := objects[m.Type.Results.List[0].Names[0]].(*Var)

	if res1 != res2 {
		t.Errorf("got %s (%p) != %s (%p)", res1, res2, res1, res2)
	}
}
