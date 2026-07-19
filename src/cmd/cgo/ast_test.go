// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func TestWalkEnum(t *testing.T) {
	const src = `package p

type Result[T any] enum {
	Ok { value T }
	None
}
`
	file, err := parser.ParseFile(token.NewFileSet(), "enum.go", src, parser.SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	var variants int
	new(File).walk(file, ctxProg, func(_ *File, x any, _ astContext) {
		if _, ok := x.(*ast.EnumVariant); ok {
			variants++
		}
	})
	if variants != 2 {
		t.Fatalf("walk visited %d variants, want 2", variants)
	}
}
