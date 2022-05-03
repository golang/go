// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source_test

import (
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/go/packages"
)

// This test reports any unexpected uses of (*go/token.File).Offset within
// the gopls codebase to ensure that we don't check in more code that is prone
// to panicking. All calls to (*go/token.File).Offset should be replaced with
// calls to source.Offset.
func TestTokenOffset(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := packages.Load(&packages.Config{
		Fset: fset,
		Mode: packages.NeedName | packages.NeedModule | packages.NeedCompiledGoFiles | packages.NeedTypes | packages.NeedTypesInfo | packages.NeedSyntax | packages.NeedImports | packages.NeedDeps,
	}, "go/token", "golang.org/x/tools/internal/lsp/...", "golang.org/x/tools/gopls/...")
	if err != nil {
		t.Fatal(err)
	}
	var tokenPkg, sourcePkg *packages.Package
	for _, pkg := range pkgs {
		switch pkg.PkgPath {
		case "go/token":
			tokenPkg = pkg
		case "golang.org/x/tools/internal/lsp/source":
			sourcePkg = pkg
		}
	}

	if tokenPkg == nil {
		t.Fatal("missing package go/token")
	}
	if sourcePkg == nil {
		t.Fatal("missing package golang.org/x/tools/internal/lsp/source")
	}

	fileObj := tokenPkg.Types.Scope().Lookup("File")
	tokenOffset, _, _ := types.LookupFieldOrMethod(fileObj.Type(), true, fileObj.Pkg(), "Offset")

	sourceOffset := sourcePkg.Types.Scope().Lookup("Offset").(*types.Func)

	for _, pkg := range pkgs {
		if pkg.PkgPath == "go/token" { // Allow usage from within go/token itself.
			continue
		}
		if pkg.PkgPath == "golang.org/x/tools/internal/lsp/lsppos" {
			continue // temporary exemption, to be refactored away
		}
		for ident, obj := range pkg.TypesInfo.Uses {
			if obj != tokenOffset {
				continue
			}
			if sourceOffset.Pos() <= ident.Pos() && ident.Pos() <= sourceOffset.Scope().End() {
				continue // accepted usage
			}
			t.Errorf(`%s: Unexpected use of (*go/token.File).Offset. Please use golang.org/x/tools/internal/lsp/source.Offset instead.`, fset.Position(ident.Pos()))
		}
	}
}
