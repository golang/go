// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"
)

func TestSearchForEnclosing(t *testing.T) {
	tests := []struct {
		desc string
		// For convenience, consider the first occurrence of the identifier "X" in
		// src.
		src string
		// By convention, "" means no type found.
		wantTypeName string
	}{
		{
			desc:         "self enclosing",
			src:          `package a; type X struct {}`,
			wantTypeName: "X",
		},
		{
			// TODO(rFindley): is this correct, or do we want to resolve I2 here?
			desc:         "embedded interface in interface",
			src:          `package a; var y = i1.X; type i1 interface {I2}; type I2 interface{X()}`,
			wantTypeName: "",
		},
		{
			desc:         "embedded interface in struct",
			src:          `package a; var y = t.X; type t struct {I}; type I interface{X()}`,
			wantTypeName: "I",
		},
		{
			desc:         "double embedding",
			src:          `package a; var y = t1.X; type t1 struct {t2}; type t2 struct {I}; type I interface{X()}`,
			wantTypeName: "I",
		},
		{
			desc:         "struct field",
			src:          `package a; type T struct { X int }`,
			wantTypeName: "T",
		},
		{
			desc:         "nested struct field",
			src:          `package a; type T struct { E struct { X int } }`,
			wantTypeName: "T",
		},
		{
			desc:         "slice entry",
			src:          `package a; type T []int; var S = T{X}; var X int = 2`,
			wantTypeName: "T",
		},
		{
			desc:         "struct pointer literal",
			src:          `package a; type T struct {i int}; var L = &T{X}; const X = 2`,
			wantTypeName: "T",
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.desc, func(t *testing.T) {
			fset := token.NewFileSet()
			file, err := parser.ParseFile(fset, "a.go", test.src, parser.AllErrors)
			if err != nil {
				t.Fatal(err)
			}
			column := 1 + bytes.IndexRune([]byte(test.src), 'X')
			pos := posAt(1, column, fset, "a.go")
			path := pathEnclosingObjNode(file, pos)
			if path == nil {
				t.Fatalf("no ident found at (1, %d)", column)
			}
			info := newInfo()
			if _, err = (*types.Config)(nil).Check("p", fset, []*ast.File{file}, info); err != nil {
				t.Fatal(err)
			}
			obj := searchForEnclosing(info, path)
			if obj == nil {
				if test.wantTypeName != "" {
					t.Errorf("searchForEnclosing(...) = <nil>, want %q", test.wantTypeName)
				}
				return
			}
			if got := obj.Name(); got != test.wantTypeName {
				t.Errorf("searchForEnclosing(...) = %q, want %q", got, test.wantTypeName)
			}
		})
	}
}

// posAt returns the token.Pos corresponding to the 1-based (line, column)
// coordinates in the file fname of fset.
func posAt(line, column int, fset *token.FileSet, fname string) token.Pos {
	var tok *token.File
	fset.Iterate(func(tf *token.File) bool {
		if tf.Name() == fname {
			tok = tf
			return false
		}
		return true
	})
	if tok == nil {
		return token.NoPos
	}
	start := tok.LineStart(line)
	return start + token.Pos(column-1)
}

// newInfo returns a types.Info with all maps populated.
func newInfo() *types.Info {
	return &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
		Scopes:     make(map[ast.Node]*types.Scope),
	}
}
