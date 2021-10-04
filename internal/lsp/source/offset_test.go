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
	var tokPkg *packages.Package
	for _, pkg := range pkgs {
		if pkg.PkgPath == "go/token" {
			tokPkg = pkg
			break
		}
	}
	typname, ok := tokPkg.Types.Scope().Lookup("File").(*types.TypeName)
	if !ok {
		t.Fatal("expected go/token.File typename, got none")
	}
	named, ok := typname.Type().(*types.Named)
	if !ok {
		t.Fatalf("expected named type, got %T", typname.Type)
	}
	var offset *types.Func
	for i := 0; i < named.NumMethods(); i++ {
		meth := named.Method(i)
		if meth.Name() == "Offset" {
			offset = meth
			break
		}
	}
	for _, pkg := range pkgs {
		for ident, obj := range pkg.TypesInfo.Uses {
			if ident.Name != "Offset" {
				continue
			}
			if pkg.PkgPath == "go/token" {
				continue
			}
			if !types.Identical(offset.Type(), obj.Type()) {
				continue
			}
			// The only permitted use is in golang.org/x/tools/internal/lsp/source.Offset,
			// so check the enclosing function.
			sourceOffset := pkg.Types.Scope().Lookup("Offset").(*types.Func)
			if sourceOffset.Pos() <= ident.Pos() && ident.Pos() <= sourceOffset.Scope().End() {
				continue // accepted usage
			}
			t.Errorf(`%s: Unexpected use of (*go/token.File).Offset. Please use golang.org/x/tools/internal/lsp/source.Offset instead.`, fset.Position(ident.Pos()))
		}
	}
}
