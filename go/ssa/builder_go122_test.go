// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.22
// +build go1.22

package ssa_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/ssa/ssautil"
)

func TestMultipleGoversions(t *testing.T) {
	var contents = map[string]string{
		"post.go": `
	//go:build go1.22
	package p

	var distinct = func(l []int) []*int {
		var r []*int
		for i := range l {
			r = append(r, &i)
		}
		return r
	}(l)
	`,
		"pre.go": `
	package p

	var l = []int{0, 0, 0}

	var same = func(l []int) []*int {
		var r []*int
		for i := range l {
			r = append(r, &i)
		}
		return r
	}(l)
	`,
	}

	fset := token.NewFileSet()
	var files []*ast.File
	for _, fname := range []string{"post.go", "pre.go"} {
		file, err := parser.ParseFile(fset, fname, contents[fname], 0)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, file)
	}

	pkg := types.NewPackage("p", "")
	conf := &types.Config{Importer: nil, GoVersion: "go1.21"}
	p, _, err := ssautil.BuildPackage(conf, fset, pkg, files, ssa.SanityCheckFunctions)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	fns := ssautil.AllFunctions(p.Prog)
	names := make(map[string]*ssa.Function)
	for fn := range fns {
		names[fn.String()] = fn
	}
	for _, item := range []struct{ name, wantSyn, wantPos string }{
		{"p.init", "package initializer", "-"},
		{"p.init$1", "", "post.go:5:17"},
		{"p.init$2", "", "pre.go:6:13"},
	} {
		fn := names[item.name]
		if fn == nil {
			t.Fatalf("Could not find function named %q in package %s", item.name, p)
		}
		if fn.Synthetic != item.wantSyn {
			t.Errorf("Function %q.Syntethic=%q. expected %q", fn, fn.Synthetic, item.wantSyn)
		}
		if got := fset.Position(fn.Pos()).String(); got != item.wantPos {
			t.Errorf("Function %q.Pos()=%q. expected %q", fn, got, item.wantPos)
		}
	}
}
