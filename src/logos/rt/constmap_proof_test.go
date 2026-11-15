// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestConstTableGetSingleProbe(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	src, err := os.ReadFile(filepath.Join(filepath.Dir(file), "constmap.go"))
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "constmap.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var get *ast.FuncDecl
	for _, d := range f.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == "Get" {
			get = fd
		}
	}
	if get == nil {
		t.Fatal("func Get not found in constmap.go")
	}
	loops, slots := 0, 0
	ast.Inspect(get.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.ForStmt, *ast.RangeStmt:
			loops++
		case *ast.IndexExpr:
			if sel, ok := n.X.(*ast.SelectorExpr); ok && sel.Sel.Name == "Slots" {
				slots++
			}
		}
		return true
	})
	if loops != 0 {
		t.Fatalf("Get holds %d loops, want none (one-probe lookup)", loops)
	}
	if slots != 1 {
		t.Fatalf("Get reads %d slot expressions, want exactly 1", slots)
	}
	for _, banned := range []string{"sort.Search", "sort.Find", "BinarySearch", "binarySearch"} {
		if strings.Contains(string(src), banned) {
			t.Fatalf("constmap.go contains %q: no logarithmic fallback may satisfy the contract", banned)
		}
	}
}
