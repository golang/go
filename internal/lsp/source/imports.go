// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"go/ast"
	"go/format"
	"go/token"
	"strconv"

	"golang.org/x/tools/internal/span"
)

// Taken and then modified from golang.org/x/tools/go/ast/astutil.
//
// We currently choose to create our own version of AddNamedImport for the following reasons:
// 	1. 	We do not want to edit the current ast. This is so that we can use the same ast
// 		to get the changes from multiple distinct modifications.
//  2.	We need the changes that *only* affect the import declarations, because the edits
// 		are not allowed to overlap with the position in the source that is being edited.
//		astutil.AddNamedImport makes it hard to determine what changes need to be made
//		to the source document from the ast, as astutil.AddNamedImport includes a merging pass.

// AddNamedImport adds the import with the given name and path to the file f, if absent.
// If name is not empty, it is used to rename the import.
//
// For example, calling
//	AddNamedImport(fset, f, "pathpkg", "path")
// adds
//	import pathpkg "path"
//
// AddNamedImport only returns edits that affect the import declarations.
func AddNamedImport(fset *token.FileSet, f *ast.File, name, path string) (edits []TextEdit, err error) {
	if alreadyImports(f, name, path) {
		return nil, nil
	}

	newImport := &ast.ImportSpec{
		Path: &ast.BasicLit{
			Kind:  token.STRING,
			Value: strconv.Quote(path),
		},
	}
	if name != "" {
		newImport.Name = &ast.Ident{Name: name}
	}

	// TODO(suzmue): insert the import statement in the location that would be chosen
	// by astutil.AddNamedImport
	// Find the last import decl.
	var lastImport = -1 // index in f.Decls of the file's final import decl
	for i, decl := range f.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if ok && gen.Tok == token.IMPORT {
			lastImport = i
		}
	}

	// Add an import decl after the last import.
	impDecl := &ast.GenDecl{
		Tok: token.IMPORT,
	}
	impDecl.Specs = append(impDecl.Specs, newImport)

	var insertPos token.Pos
	if lastImport >= 0 {
		insertPos = f.Decls[lastImport].End()
	} else {
		// There are no existing imports.
		// Our new import, preceded by a blank line,  goes after the package declaration
		// and after the comment, if any, that starts on the same line as the
		// package declaration.
		insertPos = f.Name.End()

		file := fset.File(f.Package)
		pkgLine := file.Line(f.Package)
		for _, c := range f.Comments {
			if file.Line(c.Pos()) > pkgLine {
				break
			}
			insertPos = c.End()
		}
	}

	// Print this import declaration.
	var buf bytes.Buffer
	format.Node(&buf, fset, impDecl)
	newText := "\n\n" + buf.String() + "\n"

	rng := span.NewRange(fset, insertPos, insertPos)
	spn, err := rng.Span()
	if err != nil {
		return nil, err
	}

	edits = append(edits, TextEdit{
		Span:    spn,
		NewText: newText,
	})
	return edits, nil
}

// alreadyImports reports whether f has an import with the specified name and path.
func alreadyImports(f *ast.File, name, path string) bool {
	for _, s := range f.Imports {
		if importName(s) == name && importPath(s) == path {
			return true
		}
	}
	return false
}

// importName returns the name of s,
// or "" if the import is not named.
func importName(s *ast.ImportSpec) string {
	if s.Name == nil {
		return ""
	}
	return s.Name.Name
}

// importPath returns the unquoted import path of s,
// or "" if the path is not properly quoted.
func importPath(s *ast.ImportSpec) string {
	t, err := strconv.Unquote(s.Path.Value)
	if err != nil {
		return ""
	}
	return t
}
