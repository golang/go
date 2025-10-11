// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// To avoid a cyclic dependency with go/parser, this file is in a separate package.

package ast_test

import (
	"fmt"
	. "go/ast"
	"go/parser"
	"go/token"
	"slices"
	"strings"
	"testing"
)

const src = `
// the very first comment

// package p
package p /* the name is p */

// imports
import (
	"bytes"     // bytes
	"fmt"       // fmt
	"go/ast"
	"go/parser"
)

// T
type T struct {
	a, b, c int // associated with a, b, c
	// associated with x, y
	x, y float64    // float values
	z    complex128 // complex value
}
// also associated with T

// x
var x = 0 // x = 0
// also associated with x

// f1
func f1() {
	/* associated with s1 */
	s1()
	// also associated with s1
	
	// associated with s2
	
	// also associated with s2
	s2() // line comment for s2
}
// associated with f1
// also associated with f1

// associated with f2

// f2
func f2() {
}

func f3() {
	i := 1 /* 1 */ + 2 // addition
	_ = i
}

// the very last comment
`

// res maps a key of the form "line number: node type"
// to the associated comments' text.
var res = map[string]string{
	" 5: *ast.File":       "the very first comment\npackage p\n",
	" 5: *ast.Ident":      " the name is p\n",
	" 8: *ast.GenDecl":    "imports\n",
	" 9: *ast.ImportSpec": "bytes\n",
	"10: *ast.ImportSpec": "fmt\n",
	"16: *ast.GenDecl":    "T\nalso associated with T\n",
	"17: *ast.Field":      "associated with a, b, c\n",
	"19: *ast.Field":      "associated with x, y\nfloat values\n",
	"20: *ast.Field":      "complex value\n",
	"25: *ast.GenDecl":    "x\nx = 0\nalso associated with x\n",
	"29: *ast.FuncDecl":   "f1\nassociated with f1\nalso associated with f1\n",
	"31: *ast.ExprStmt":   " associated with s1\nalso associated with s1\n",
	"37: *ast.ExprStmt":   "associated with s2\nalso associated with s2\nline comment for s2\n",
	"45: *ast.FuncDecl":   "associated with f2\nf2\n",
	"49: *ast.AssignStmt": "addition\n",
	"49: *ast.BasicLit":   " 1\n",
	"50: *ast.Ident":      "the very last comment\n",
}

func ctext(list []*CommentGroup) string {
	var buf strings.Builder
	for _, g := range list {
		buf.WriteString(g.Text())
	}
	return buf.String()
}

func TestCommentMap(t *testing.T) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	cmap := NewCommentMap(fset, f, f.Comments)

	// very correct association of comments
	for n, list := range cmap {
		key := fmt.Sprintf("%2d: %T", fset.Position(n.Pos()).Line, n)
		got := ctext(list)
		want := res[key]
		if got != want {
			t.Errorf("%s: got %q; want %q", key, got, want)
		}
	}

	// verify that no comments got lost
	if n := len(cmap.Comments()); n != len(f.Comments) {
		t.Errorf("got %d comment groups in map; want %d", n, len(f.Comments))
	}

	// support code to update test:
	// set genMap to true to generate res map
	const genMap = false
	if genMap {
		out := make([]string, 0, len(cmap))
		for n, list := range cmap {
			out = append(out, fmt.Sprintf("\t\"%2d: %T\":\t%q,", fset.Position(n.Pos()).Line, n, ctext(list)))
		}
		slices.Sort(out)
		for _, s := range out {
			fmt.Println(s)
		}
	}
}

func TestFilter(t *testing.T) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	cmap := NewCommentMap(fset, f, f.Comments)

	// delete variable declaration
	for i, decl := range f.Decls {
		if gen, ok := decl.(*GenDecl); ok && gen.Tok == token.VAR {
			copy(f.Decls[i:], f.Decls[i+1:])
			f.Decls = f.Decls[:len(f.Decls)-1]
			break
		}
	}

	// check if comments are filtered correctly
	cc := cmap.Filter(f)
	for n, list := range cc {
		key := fmt.Sprintf("%2d: %T", fset.Position(n.Pos()).Line, n)
		got := ctext(list)
		want := res[key]
		if key == "25: *ast.GenDecl" || got != want {
			t.Errorf("%s: got %q; want %q", key, got, want)
		}
	}
}
