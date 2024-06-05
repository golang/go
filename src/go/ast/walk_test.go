// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func TestPreorderBreak(t *testing.T) {
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
