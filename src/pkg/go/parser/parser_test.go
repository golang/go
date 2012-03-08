// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"testing"
)

var fset = token.NewFileSet()

var validFiles = []string{
	"parser.go",
	"parser_test.go",
	"error_test.go",
	"short_test.go",
}

func TestParse(t *testing.T) {
	for _, filename := range validFiles {
		_, err := ParseFile(fset, filename, nil, DeclarationErrors)
		if err != nil {
			t.Errorf("ParseFile(%s): %v", filename, err)
		}
	}
}

func nameFilter(filename string) bool {
	switch filename {
	case "parser.go":
	case "interface.go":
	case "parser_test.go":
	default:
		return false
	}
	return true
}

func dirFilter(f os.FileInfo) bool { return nameFilter(f.Name()) }

func TestParseDir(t *testing.T) {
	path := "."
	pkgs, err := ParseDir(fset, path, dirFilter, 0)
	if err != nil {
		t.Fatalf("ParseDir(%s): %v", path, err)
	}
	if len(pkgs) != 1 {
		t.Errorf("incorrect number of packages: %d", len(pkgs))
	}
	pkg := pkgs["parser"]
	if pkg == nil {
		t.Errorf(`package "parser" not found`)
		return
	}
	for filename := range pkg.Files {
		if !nameFilter(filename) {
			t.Errorf("unexpected package file: %s", filename)
		}
	}
}

func TestParseExpr(t *testing.T) {
	// just kicking the tires:
	// a valid expression
	src := "a + b"
	x, err := ParseExpr(src)
	if err != nil {
		t.Errorf("ParseExpr(%s): %v", src, err)
	}
	// sanity check
	if _, ok := x.(*ast.BinaryExpr); !ok {
		t.Errorf("ParseExpr(%s): got %T, expected *ast.BinaryExpr", src, x)
	}

	// an invalid expression
	src = "a + *"
	_, err = ParseExpr(src)
	if err == nil {
		t.Errorf("ParseExpr(%s): %v", src, err)
	}

	// it must not crash
	for _, src := range valids {
		ParseExpr(src)
	}
}

func TestColonEqualsScope(t *testing.T) {
	f, err := ParseFile(fset, "", `package p; func f() { x, y, z := x, y, z }`, 0)
	if err != nil {
		t.Errorf("parse: %s", err)
	}

	// RHS refers to undefined globals; LHS does not.
	as := f.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.AssignStmt)
	for _, v := range as.Rhs {
		id := v.(*ast.Ident)
		if id.Obj != nil {
			t.Errorf("rhs %s has Obj, should not", id.Name)
		}
	}
	for _, v := range as.Lhs {
		id := v.(*ast.Ident)
		if id.Obj == nil {
			t.Errorf("lhs %s does not have Obj, should", id.Name)
		}
	}
}

func TestVarScope(t *testing.T) {
	f, err := ParseFile(fset, "", `package p; func f() { var x, y, z = x, y, z }`, 0)
	if err != nil {
		t.Errorf("parse: %s", err)
	}

	// RHS refers to undefined globals; LHS does not.
	as := f.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.DeclStmt).Decl.(*ast.GenDecl).Specs[0].(*ast.ValueSpec)
	for _, v := range as.Values {
		id := v.(*ast.Ident)
		if id.Obj != nil {
			t.Errorf("rhs %s has Obj, should not", id.Name)
		}
	}
	for _, id := range as.Names {
		if id.Obj == nil {
			t.Errorf("lhs %s does not have Obj, should", id.Name)
		}
	}
}

var imports = map[string]bool{
	`"a"`:        true,
	"`a`":        true,
	`"a/b"`:      true,
	`"a.b"`:      true,
	`"m\x61th"`:  true,
	`"greek/αβ"`: true,
	`""`:         false,

	// Each of these pairs tests both `` vs "" strings
	// and also use of invalid characters spelled out as
	// escape sequences and written directly.
	// For example `"\x00"` tests import "\x00"
	// while "`\x00`" tests import `<actual-NUL-byte>`.
	`"\x00"`:     false,
	"`\x00`":     false,
	`"\x7f"`:     false,
	"`\x7f`":     false,
	`"a!"`:       false,
	"`a!`":       false,
	`"a b"`:      false,
	"`a b`":      false,
	`"a\\b"`:     false,
	"`a\\b`":     false,
	"\"`a`\"":    false,
	"`\"a\"`":    false,
	`"\x80\x80"`: false,
	"`\x80\x80`": false,
	`"\xFFFD"`:   false,
	"`\xFFFD`":   false,
}

func TestImports(t *testing.T) {
	for path, isValid := range imports {
		src := fmt.Sprintf("package p; import %s", path)
		_, err := ParseFile(fset, "", src, 0)
		switch {
		case err != nil && isValid:
			t.Errorf("ParseFile(%s): got %v; expected no error", src, err)
		case err == nil && !isValid:
			t.Errorf("ParseFile(%s): got no error; expected one", src)
		}
	}
}
