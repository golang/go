// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.20
// +build go1.20

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

func TestBuildPackageGo120(t *testing.T) {
	tests := []struct {
		name     string
		src      string
		importer types.Importer
	}{
		{"slice to array", "package p; var s []byte; var _ = ([4]byte)(s)", nil},
		{"slice to zero length array", "package p; var s []byte; var _ = ([0]byte)(s)", nil},
		{"slice to zero length array type parameter", "package p; var s []byte; func f[T ~[0]byte]() { tmp := (T)(s); var z T; _ = tmp == z}", nil},
		{"slice to non-zero length array type parameter", "package p; var s []byte; func h[T ~[1]byte | [4]byte]() { tmp := T(s); var z T; _ = tmp == z}", nil},
		{"slice to maybe-zero length array type parameter", "package p; var s []byte; func g[T ~[0]byte | [4]byte]() { tmp := T(s); var z T; _ = tmp == z}", nil},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "p.go", tc.src, parser.ParseComments)
			if err != nil {
				t.Error(err)
			}
			files := []*ast.File{f}

			pkg := types.NewPackage("p", "")
			conf := &types.Config{Importer: tc.importer}
			if _, _, err := ssautil.BuildPackage(conf, fset, pkg, files, ssa.SanityCheckFunctions); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}
