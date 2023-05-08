// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetoken_test

import (
	"fmt"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/internal/testenv"
)

func TestWorkaroundIssue57490(t *testing.T) {
	// During error recovery the parser synthesizes various close
	// tokens at EOF, causing the End position of incomplete
	// syntax nodes, computed as Rbrace+len("}"), to be beyond EOF.
	src := `package p; func f() { var x struct`
	fset := token.NewFileSet()
	file, _ := parser.ParseFile(fset, "a.go", src, 0)
	tf := fset.File(file.Pos())

	// Add another file to the FileSet.
	file2, _ := parser.ParseFile(fset, "b.go", "package q", 0)

	// This is the ambiguity of #57490...
	if file.End() != file2.Pos() {
		t.Errorf("file.End() %d != %d file2.Pos()", file.End(), file2.Pos())
	}
	// ...which causes these statements to panic.
	if false {
		tf.Offset(file.End())   // panic: invalid Pos value 36 (should be in [1, 35])
		tf.Position(file.End()) // panic: invalid Pos value 36 (should be in [1, 35])
	}

	// The offset of the EOF position is the file size.
	offset, err := safetoken.Offset(tf, file.End()-1)
	if err != nil || offset != tf.Size() {
		t.Errorf("Offset(EOF) = (%d, %v), want token.File.Size %d", offset, err, tf.Size())
	}

	// The offset of the file.End() position, 1 byte beyond EOF,
	// is also the size of the file.
	offset, err = safetoken.Offset(tf, file.End())
	if err != nil || offset != tf.Size() {
		t.Errorf("Offset(ast.File.End()) = (%d, %v), want token.File.Size %d", offset, err, tf.Size())
	}

	if got, want := safetoken.Position(tf, file.End()).String(), "a.go:1:35"; got != want {
		t.Errorf("Position(ast.File.End()) = %s, want %s", got, want)
	}

	if got, want := safetoken.EndPosition(fset, file.End()).String(), "a.go:1:35"; got != want {
		t.Errorf("EndPosition(ast.File.End()) = %s, want %s", got, want)
	}

	// Note that calling StartPosition on an end may yield the wrong file:
	if got, want := safetoken.StartPosition(fset, file.End()).String(), "b.go:1:1"; got != want {
		t.Errorf("StartPosition(ast.File.End()) = %s, want %s", got, want)
	}
}

// To reduce the risk of panic, or bugs for which this package
// provides a workaround, this test statically reports references to
// forbidden methods of token.File or FileSet throughout gopls and
// suggests alternatives.
func TestGoplsSourceDoesNotCallTokenFileMethods(t *testing.T) {
	testenv.NeedsGoPackages(t)
	testenv.NeedsGo1Point(t, 18)
	testenv.NeedsLocalXTools(t)

	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedModule | packages.NeedCompiledGoFiles | packages.NeedTypes | packages.NeedTypesInfo | packages.NeedSyntax | packages.NeedImports | packages.NeedDeps,
	}
	cfg.Env = os.Environ()
	cfg.Env = append(cfg.Env,
		"GOPACKAGESDRIVER=off",
		"GOWORK=off", // necessary for -mod=mod below
		"GOFLAGS=-mod=mod",
	)

	pkgs, err := packages.Load(cfg, "go/token", "golang.org/x/tools/gopls/...")
	if err != nil {
		t.Fatal(err)
	}
	var tokenPkg *packages.Package
	for _, pkg := range pkgs {
		if pkg.PkgPath == "go/token" {
			tokenPkg = pkg
			break
		}
	}
	if tokenPkg == nil {
		t.Fatal("missing package go/token")
	}

	File := tokenPkg.Types.Scope().Lookup("File")
	FileSet := tokenPkg.Types.Scope().Lookup("FileSet")

	alternative := make(map[types.Object]string)
	setAlternative := func(recv types.Object, old, new string) {
		oldMethod, _, _ := types.LookupFieldOrMethod(recv.Type(), true, recv.Pkg(), old)
		alternative[oldMethod] = new
	}
	setAlternative(File, "Line", "safetoken.Line")
	setAlternative(File, "Offset", "safetoken.Offset")
	setAlternative(File, "Position", "safetoken.Position")
	setAlternative(File, "PositionFor", "safetoken.Position")
	setAlternative(FileSet, "Position", "safetoken.StartPosition or EndPosition")
	setAlternative(FileSet, "PositionFor", "safetoken.StartPosition or EndPosition")

	for _, pkg := range pkgs {
		switch pkg.PkgPath {
		case "go/token", "golang.org/x/tools/gopls/internal/lsp/safetoken":
			continue // allow calls within these packages
		}

		for ident, obj := range pkg.TypesInfo.Uses {
			if alt, ok := alternative[obj]; ok {
				posn := safetoken.StartPosition(pkg.Fset, ident.Pos())
				fmt.Fprintf(os.Stderr, "%s: forbidden use of %v; use %s instead.\n", posn, obj, alt)
				t.Fail()
			}
		}
	}
}
