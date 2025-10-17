// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package refactor

// This file defines operations for computing edits to imports.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	pathpkg "path"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/packagepath"
)

// AddImport returns the prefix (either "pkg." or "") that should be
// used to qualify references to the desired symbol (member) imported
// from the specified package, plus any necessary edits to the file's
// import declaration to add a new import.
//
// If the import already exists, and is accessible at pos, AddImport
// returns the existing name and no edits. (If the existing import is
// a dot import, the prefix is "".)
//
// Otherwise, it adds a new import, using a local name derived from
// the preferred name. To request a blank import, use a preferredName
// of "_", and discard the prefix result; member is ignored in this
// case.
//
// AddImport accepts the caller's implicit claim that the imported
// package declares member.
//
// AddImport does not mutate its arguments.
func AddImport(info *types.Info, file *ast.File, preferredName, pkgpath, member string, pos token.Pos) (prefix string, edits []analysis.TextEdit) {
	// Find innermost enclosing lexical block.
	scope := info.Scopes[file].Innermost(pos)
	if scope == nil {
		panic("no enclosing lexical block")
	}

	// Is there an existing import of this package?
	// If so, are we in its scope? (not shadowed)
	for _, spec := range file.Imports {
		pkgname := info.PkgNameOf(spec)
		if pkgname != nil && pkgname.Imported().Path() == pkgpath {
			name := pkgname.Name()
			if preferredName == "_" {
				// Request for blank import; any existing import will do.
				return "", nil
			}
			if name == "." {
				// The scope of ident must be the file scope.
				if s, _ := scope.LookupParent(member, pos); s == info.Scopes[file] {
					return "", nil
				}
			} else if _, obj := scope.LookupParent(name, pos); obj == pkgname {
				return name + ".", nil
			}
		}
	}

	// We must add a new import.

	// Ensure we have a fresh name.
	newName := preferredName
	if preferredName != "_" {
		newName = FreshName(scope, pos, preferredName)
	}

	// Create a new import declaration either before the first existing
	// declaration (which must exist), including its comments; or
	// inside the declaration, if it is an import group.
	//
	// Use a renaming import whenever the preferred name is not
	// available, or the chosen name does not match the last
	// segment of its path.
	newText := fmt.Sprintf("%q", pkgpath)
	if newName != preferredName || newName != pathpkg.Base(pkgpath) {
		newText = fmt.Sprintf("%s %q", newName, pkgpath)
	}

	decl0 := file.Decls[0]
	var before ast.Node = decl0
	switch decl0 := decl0.(type) {
	case *ast.GenDecl:
		if decl0.Doc != nil {
			before = decl0.Doc
		}
	case *ast.FuncDecl:
		if decl0.Doc != nil {
			before = decl0.Doc
		}
	}
	if gd, ok := before.(*ast.GenDecl); ok && gd.Tok == token.IMPORT && gd.Rparen.IsValid() {
		// Have existing grouped import ( ... ) decl.
		if packagepath.IsStdPackage(pkgpath) && len(gd.Specs) > 0 {
			// Add spec for a std package before
			// first existing spec, followed by
			// a blank line if the next one is non-std.
			first := gd.Specs[0].(*ast.ImportSpec)
			pos = first.Pos()
			if !packagepath.IsStdPackage(first.Path.Value) {
				newText += "\n"
			}
			newText += "\n\t"
		} else {
			// Add spec at end of group.
			pos = gd.Rparen
			newText = "\t" + newText + "\n"
		}
	} else {
		// No import decl, or non-grouped import.
		// Add a new import decl before first decl.
		// (gofmt will merge multiple import decls.)
		pos = before.Pos()
		newText = "import " + newText + "\n\n"
	}
	return newName + ".", []analysis.TextEdit{{
		Pos:     pos,
		End:     pos,
		NewText: []byte(newText),
	}}
}
