// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inspector_test

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"log"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams"
)

var netFiles []*ast.File

func init() {
	files, err := parseNetFiles()
	if err != nil {
		log.Fatal(err)
	}
	netFiles = files
}

func parseNetFiles() ([]*ast.File, error) {
	pkg, err := build.Default.Import("net", "", 0)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	var files []*ast.File
	for _, filename := range pkg.GoFiles {
		filename = filepath.Join(pkg.Dir, filename)
		f, err := parser.ParseFile(fset, filename, nil, 0)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	return files, nil
}

// TestAllNodes compares Inspector against ast.Inspect.
func TestInspectAllNodes(t *testing.T) {
	inspect := inspector.New(netFiles)

	var nodesA []ast.Node
	inspect.Nodes(nil, func(n ast.Node, push bool) bool {
		if push {
			nodesA = append(nodesA, n)
		}
		return true
	})
	var nodesB []ast.Node
	for _, f := range netFiles {
		ast.Inspect(f, func(n ast.Node) bool {
			if n != nil {
				nodesB = append(nodesB, n)
			}
			return true
		})
	}
	compare(t, nodesA, nodesB)
}

func TestInspectGenericNodes(t *testing.T) {
	if !typeparams.Enabled {
		t.Skip("type parameters are not supported at this Go version")
	}

	// src is using the 16 identifiers i0, i1, ... i15 so
	// we can easily verify that we've found all of them.
	const src = `package a

type I interface { ~i0|i1 }

type T[i2, i3 interface{ ~i4 }] struct {}

func f[i5, i6 any]() {
	_ = f[i7, i8]
	var x T[i9, i10]
}

func (*T[i11, i12]) m()

var _ i13[i14, i15]
`
	fset := token.NewFileSet()
	f, _ := parser.ParseFile(fset, "a.go", src, 0)
	inspect := inspector.New([]*ast.File{f})
	found := make([]bool, 16)

	indexListExprs := make(map[*typeparams.IndexListExpr]bool)

	// Verify that we reach all i* identifiers, and collect IndexListExpr nodes.
	inspect.Preorder(nil, func(n ast.Node) {
		switch n := n.(type) {
		case *ast.Ident:
			if n.Name[0] == 'i' {
				index, err := strconv.Atoi(n.Name[1:])
				if err != nil {
					t.Fatal(err)
				}
				found[index] = true
			}
		case *typeparams.IndexListExpr:
			indexListExprs[n] = false
		}
	})
	for i, v := range found {
		if !v {
			t.Errorf("missed identifier i%d", i)
		}
	}

	// Verify that we can filter to IndexListExprs that we found in the first
	// step.
	if len(indexListExprs) == 0 {
		t.Fatal("no index list exprs found")
	}
	inspect.Preorder([]ast.Node{&typeparams.IndexListExpr{}}, func(n ast.Node) {
		ix := n.(*typeparams.IndexListExpr)
		indexListExprs[ix] = true
	})
	for ix, v := range indexListExprs {
		if !v {
			t.Errorf("inspected node %v not filtered", ix)
		}
	}
}

// TestPruning compares Inspector against ast.Inspect,
// pruning descent within ast.CallExpr nodes.
func TestInspectPruning(t *testing.T) {
	inspect := inspector.New(netFiles)

	var nodesA []ast.Node
	inspect.Nodes(nil, func(n ast.Node, push bool) bool {
		if push {
			nodesA = append(nodesA, n)
			_, isCall := n.(*ast.CallExpr)
			return !isCall // don't descend into function calls
		}
		return false
	})
	var nodesB []ast.Node
	for _, f := range netFiles {
		ast.Inspect(f, func(n ast.Node) bool {
			if n != nil {
				nodesB = append(nodesB, n)
				_, isCall := n.(*ast.CallExpr)
				return !isCall // don't descend into function calls
			}
			return false
		})
	}
	compare(t, nodesA, nodesB)
}

func compare(t *testing.T, nodesA, nodesB []ast.Node) {
	if len(nodesA) != len(nodesB) {
		t.Errorf("inconsistent node lists: %d vs %d", len(nodesA), len(nodesB))
	} else {
		for i := range nodesA {
			if a, b := nodesA[i], nodesB[i]; a != b {
				t.Errorf("node %d is inconsistent: %T, %T", i, a, b)
			}
		}
	}
}

func TestTypeFiltering(t *testing.T) {
	const src = `package a
func f() {
	print("hi")
	panic("oops")
}
`
	fset := token.NewFileSet()
	f, _ := parser.ParseFile(fset, "a.go", src, 0)
	inspect := inspector.New([]*ast.File{f})

	var got []string
	fn := func(n ast.Node, push bool) bool {
		if push {
			got = append(got, typeOf(n))
		}
		return true
	}

	// no type filtering
	inspect.Nodes(nil, fn)
	if want := strings.Fields("File Ident FuncDecl Ident FuncType FieldList BlockStmt ExprStmt CallExpr Ident BasicLit ExprStmt CallExpr Ident BasicLit"); !reflect.DeepEqual(got, want) {
		t.Errorf("inspect: got %s, want %s", got, want)
	}

	// type filtering
	nodeTypes := []ast.Node{
		(*ast.BasicLit)(nil),
		(*ast.CallExpr)(nil),
	}
	got = nil
	inspect.Nodes(nodeTypes, fn)
	if want := strings.Fields("CallExpr BasicLit CallExpr BasicLit"); !reflect.DeepEqual(got, want) {
		t.Errorf("inspect: got %s, want %s", got, want)
	}

	// inspect with stack
	got = nil
	inspect.WithStack(nodeTypes, func(n ast.Node, push bool, stack []ast.Node) bool {
		if push {
			var line []string
			for _, n := range stack {
				line = append(line, typeOf(n))
			}
			got = append(got, strings.Join(line, " "))
		}
		return true
	})
	want := []string{
		"File FuncDecl BlockStmt ExprStmt CallExpr",
		"File FuncDecl BlockStmt ExprStmt CallExpr BasicLit",
		"File FuncDecl BlockStmt ExprStmt CallExpr",
		"File FuncDecl BlockStmt ExprStmt CallExpr BasicLit",
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("inspect: got %s, want %s", got, want)
	}
}

func typeOf(n ast.Node) string {
	return strings.TrimPrefix(reflect.TypeOf(n).String(), "*ast.")
}

// The numbers show a marginal improvement (ASTInspect/Inspect) of 3.5x,
// but a break-even point (NewInspector/(ASTInspect-Inspect)) of about 5
// traversals.
//
// BenchmarkASTInspect     1.0 ms
// BenchmarkNewInspector   2.2 ms
// BenchmarkInspect        0.39ms
// BenchmarkInspectFilter  0.01ms
// BenchmarkInspectCalls   0.14ms

func BenchmarkNewInspector(b *testing.B) {
	// Measure one-time construction overhead.
	for i := 0; i < b.N; i++ {
		inspector.New(netFiles)
	}
}

func BenchmarkInspect(b *testing.B) {
	b.StopTimer()
	inspect := inspector.New(netFiles)
	b.StartTimer()

	// Measure marginal cost of traversal.
	var ndecls, nlits int
	for i := 0; i < b.N; i++ {
		inspect.Preorder(nil, func(n ast.Node) {
			switch n.(type) {
			case *ast.FuncDecl:
				ndecls++
			case *ast.FuncLit:
				nlits++
			}
		})
	}
}

func BenchmarkInspectFilter(b *testing.B) {
	b.StopTimer()
	inspect := inspector.New(netFiles)
	b.StartTimer()

	// Measure marginal cost of traversal.
	nodeFilter := []ast.Node{(*ast.FuncDecl)(nil), (*ast.FuncLit)(nil)}
	var ndecls, nlits int
	for i := 0; i < b.N; i++ {
		inspect.Preorder(nodeFilter, func(n ast.Node) {
			switch n.(type) {
			case *ast.FuncDecl:
				ndecls++
			case *ast.FuncLit:
				nlits++
			}
		})
	}
}

func BenchmarkInspectCalls(b *testing.B) {
	b.StopTimer()
	inspect := inspector.New(netFiles)
	b.StartTimer()

	// Measure marginal cost of traversal.
	nodeFilter := []ast.Node{(*ast.CallExpr)(nil)}
	var ncalls int
	for i := 0; i < b.N; i++ {
		inspect.Preorder(nodeFilter, func(n ast.Node) {
			_ = n.(*ast.CallExpr)
			ncalls++
		})
	}
}

func BenchmarkASTInspect(b *testing.B) {
	var ndecls, nlits int
	for i := 0; i < b.N; i++ {
		for _, f := range netFiles {
			ast.Inspect(f, func(n ast.Node) bool {
				switch n.(type) {
				case *ast.FuncDecl:
					ndecls++
				case *ast.FuncLit:
					nlits++
				}
				return true
			})
		}
	}
}
