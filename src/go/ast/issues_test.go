// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func TestIssue33649(t *testing.T) {
	for _, src := range []string{
		`package p; func _()`,
		`package p; func _() {`,
		`package p; func _() { _ = 0`,
		`package p; func _() { _ = 0 }`,
	} {
		fset := token.NewFileSet()
		f, _ := parser.ParseFile(fset, "", src, parser.AllErrors)
		if f == nil {
			panic("invalid test setup: parser didn't return an AST")
		}

		// find corresponding token.File
		var tf *token.File
		fset.Iterate(func(f *token.File) bool {
			tf = f
			return true
		})
		tfEnd := tf.Base() + tf.Size()

		fd := f.Decls[len(f.Decls)-1].(*ast.FuncDecl)
		fdEnd := int(fd.End())

		if fdEnd != tfEnd {
			t.Errorf("%q: got fdEnd = %d; want %d (base = %d, size = %d)", src, fdEnd, tfEnd, tf.Base(), tf.Size())
		}
	}
}
