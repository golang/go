// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package typeutil_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	"golang.org/x/tools/go/types"
	"golang.org/x/tools/go/types/typeutil"
)

func TestDependencies(t *testing.T) {
	packages := make(map[string]*types.Package)
	conf := types.Config{
		Packages: packages,
		Import: func(_ map[string]*types.Package, path string) (*types.Package, error) {
			return packages[path], nil
		},
	}
	fset := token.NewFileSet()

	// All edges go to the right.
	//  /--D--B--A
	// F    \_C_/
	//  \__E_/
	for i, content := range []string{
		`package a`,
		`package c; import (_ "a")`,
		`package b; import (_ "a")`,
		`package e; import (_ "c")`,
		`package d; import (_ "b"; _ "c")`,
		`package f; import (_ "d"; _ "e")`,
	} {
		f, err := parser.ParseFile(fset, fmt.Sprintf("%d.go", i), content, 0)
		if err != nil {
			t.Fatal(err)
		}
		pkg, err := conf.Check(f.Name.Name, fset, []*ast.File{f}, nil)
		if err != nil {
			t.Fatal(err)
		}
		packages[pkg.Path()] = pkg
	}

	for _, test := range []struct {
		roots, want string
	}{
		{"a", "a"},
		{"b", "ab"},
		{"c", "ac"},
		{"d", "abcd"},
		{"e", "ace"},
		{"f", "abcdef"},

		{"be", "abce"},
		{"eb", "aceb"},
		{"de", "abcde"},
		{"ed", "acebd"},
		{"ef", "acebdf"},
	} {
		var pkgs []*types.Package
		for _, r := range test.roots {
			pkgs = append(pkgs, packages[string(r)])
		}
		var got string
		for _, p := range typeutil.Dependencies(pkgs...) {
			got += p.Path()
		}
		if got != test.want {
			t.Errorf("Dependencies(%q) = %q, want %q", test.roots, got, test.want)
		}
	}
}
