// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"slices"
	"strings"
	"testing"
)

func TestPreorder_Break(t *testing.T) {
	// This test checks that Preorder correctly handles a break statement while
	// in the middle of walking a node. Previously, incorrect handling of the
	// boolean returned by the yield function resulted in the iterator calling
	// yield for sibling nodes even after yield had returned false. With that
	// bug, this test failed with a runtime panic.
	src := "package p\ntype T struct {\n\tF int `json:\"f\"` // a field\n}\n"

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, 0)
	if err != nil {
		panic(err)
	}

	for n := range ast.Preorder(f) {
		if id, ok := n.(*ast.Ident); ok && id.Name == "F" {
			break
		}
	}
}

func TestPreorderStack(t *testing.T) {
	const src = `package a
func f() {
	print("hello")
}
func g() {
	print("goodbye")
	panic("oops")
}
`
	fset := token.NewFileSet()
	f, _ := parser.ParseFile(fset, "a.go", src, 0)

	str := func(n ast.Node) string {
		return strings.TrimPrefix(reflect.TypeOf(n).String(), "*ast.")
	}

	var events []string
	var gotStack []string
	ast.PreorderStack(f, nil, func(n ast.Node, stack []ast.Node) bool {
		events = append(events, str(n))
		if decl, ok := n.(*ast.FuncDecl); ok && decl.Name.Name == "f" {
			return false // skip subtree of f()
		}
		if lit, ok := n.(*ast.BasicLit); ok && lit.Value == `"oops"` {
			for _, n := range stack {
				gotStack = append(gotStack, str(n))
			}
		}
		return true
	})

	// Check sequence of events.
	wantEvents := []string{
		"File", "Ident", // package a
		"FuncDecl",                                                // func f()  [pruned]
		"FuncDecl", "Ident", "FuncType", "FieldList", "BlockStmt", // func g()
		"ExprStmt", "CallExpr", "Ident", "BasicLit", // print...
		"ExprStmt", "CallExpr", "Ident", "BasicLit", // panic...
	}
	if !slices.Equal(events, wantEvents) {
		t.Errorf("PreorderStack events:\ngot:  %s\nwant: %s", events, wantEvents)
	}

	// Check captured stack.
	wantStack := []string{"File", "FuncDecl", "BlockStmt", "ExprStmt", "CallExpr"}
	if !slices.Equal(gotStack, wantStack) {
		t.Errorf("PreorderStack stack:\ngot:  %s\nwant: %s", gotStack, wantStack)
	}
}
