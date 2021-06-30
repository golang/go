// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.17
// +build go1.17

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

func TestSliceToArrayPtr(t *testing.T) {
	src := `package p

func f() {
	var s []byte
	_ = (*[4]byte)(s)
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	files := []*ast.File{f}

	pkg := types.NewPackage("p", "")
	conf := &types.Config{}
	if _, _, err := ssautil.BuildPackage(conf, fset, pkg, files, ssa.SanityCheckFunctions); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}
