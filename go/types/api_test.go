// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(gri) This file needs to be expanded significantly.

package types

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func pkgFor(path string, source string, info *Info) (*Package, error) {
	fset = token.NewFileSet()
	f, err := parser.ParseFile(fset, path, source, 0)
	if err != nil {
		return nil, err
	}

	var conf Config
	pkg, err := conf.Check(path, fset, []*ast.File{f}, info)
	if err != nil {
		return nil, err
	}

	return pkg, nil
}

func TestCommaOkTypes(t *testing.T) {
	var tests = []struct {
		src  string
		expr string // comma-ok expression string
		typ  string // typestring of comma-ok value
	}{
		{`package p; var x interface{}; var _, _ = x.(int)`,
			`x.(int)`,
			`(int, bool)`,
		},
		{`package p; var x interface{}; func _() { _, _ = x.(int) }`,
			`x.(int)`,
			`(int, bool)`,
		},
		{`package p; type mybool bool; var m map[string]complex128; var b mybool; func _() { _, b = m["foo"] }`,
			`m["foo"]`,
			`(complex128, p.mybool)`,
		},
		{`package p; var c chan string; var _, _ = <-c`,
			`<-c`,
			`(string, bool)`,
		},
	}

	for i, test := range tests {
		path := fmt.Sprintf("CommaOk%d", i)

		// type-check
		info := Info{Types: make(map[ast.Expr]Type)}
		_, err := pkgFor(path, test.src, &info)
		if err != nil {
			t.Error(err)
			continue
		}

		// look for comma-ok expression type
		var typ Type
		for e, t := range info.Types {
			if exprString(e) == test.expr {
				typ = t
				break
			}
		}
		if typ == nil {
			t.Errorf("%s: no type found for %s", path, test.expr)
			continue
		}

		// check that type is correct
		if got := typ.String(); got != test.typ {
			t.Errorf("%s: got %s; want %s", path, got, test.typ)
		}
	}
}
