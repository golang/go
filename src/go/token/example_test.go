// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
)

func Example_retrievePositionInfo() {
	fset := token.NewFileSet()

	const src = `package main

import "fmt"

import "go/token"

//line :1:5
type p = token.Pos

const bad = token.NoPos

//line fake.go:42:11
func ok(pos p) bool {
	return pos != bad
}

/*line :7:9*/func main() {
	fmt.Println(ok(bad) == bad.IsValid())
}
`

	f, err := parser.ParseFile(fset, "main.go", src, 0)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Print the location and kind of each declaration in f.
	for _, decl := range f.Decls {
		// Get the filename, line, and column back via the file set.
		// We get both the relative and absolute position.
		// The relative position is relative to the last line directive.
		// The absolute position is the exact position in the source.
		pos := decl.Pos()
		relPosition := fset.Position(pos)
		absPosition := fset.PositionFor(pos, false)

		// Either a FuncDecl or GenDecl, since we exit on error.
		kind := "func"
		if gen, ok := decl.(*ast.GenDecl); ok {
			kind = gen.Tok.String()
		}

		// If the relative and absolute positions differ, show both.
		fmtPosition := relPosition.String()
		if relPosition != absPosition {
			fmtPosition += "[" + absPosition.String() + "]"
		}

		fmt.Printf("%s: %s\n", fmtPosition, kind)
	}

	// Output:
	//
	// main.go:3:1: import
	// main.go:5:1: import
	// main.go:1:5[main.go:8:1]: type
	// main.go:3:1[main.go:10:1]: const
	// fake.go:42:11[main.go:13:1]: func
	// fake.go:7:9[main.go:17:14]: func
}
