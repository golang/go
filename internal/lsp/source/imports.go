// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"strconv"
	"strings"

	"golang.org/x/tools/internal/lsp/diff"
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

// addNamedImport adds the import with the given name and path to the file f, if absent.
// If name is not empty, it is used to rename the import.
//
// For example, calling
//	addNamedImport(fset, f, "pathpkg", "path")
// adds
//	import pathpkg "path"
//
// addNamedImport only returns edits that affect the import declarations.
func addNamedImport(fset *token.FileSet, f *ast.File, name, path string) (edits []diff.TextEdit, err error) {
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

	// Find an import decl to add to.
	// The goal is to find an existing import
	// whose import path has the longest shared
	// prefix with path.
	var (
		bestMatch  = -1         // length of longest shared prefix
		lastImport = -1         // index in f.Decls of the file's final import decl
		impDecl    *ast.GenDecl // import decl containing the best match
		impIndex   = -1         // spec index in impDecl containing the best match

		isThirdPartyPath = isThirdParty(path)
	)
	for i, decl := range f.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if ok && gen.Tok == token.IMPORT {
			lastImport = i
			// Do not add to import "C", to avoid disrupting the
			// association with its doc comment, breaking cgo.
			if declImports(gen, "C") {
				continue
			}

			// Do not add to single imports.
			if !gen.Lparen.IsValid() {
				continue
			}

			// Match an empty import decl if that's all that is available.
			if len(gen.Specs) == 0 && bestMatch == -1 {
				impDecl = gen
			}

			// Compute longest shared prefix with imports in this group and find best
			// matched import spec.
			// 1. Always prefer import spec with longest shared prefix.
			// 2. While match length is 0,
			// - for stdlib package: prefer first import spec.
			// - for third party package: prefer first third party import spec.
			// We cannot use last import spec as best match for third party package
			// because grouped imports are usually placed last by goimports -local
			// flag.
			// See issue #19190.
			seenAnyThirdParty := false
			for j, spec := range gen.Specs {
				impspec := spec.(*ast.ImportSpec)
				p := importPath(impspec)
				n := matchLen(p, path)
				if n > bestMatch || (bestMatch == 0 && !seenAnyThirdParty && isThirdPartyPath) {
					bestMatch = n
					impDecl = gen
					impIndex = j
				}
				seenAnyThirdParty = seenAnyThirdParty || isThirdParty(p)
			}
		}
	}

	var insertPos token.Pos
	var newText string
	// If no import decl found, add one after the last import.
	if impDecl == nil {
		// Add an import decl after the last import.
		impDecl = &ast.GenDecl{
			Tok: token.IMPORT,
		}
		impDecl.Specs = append(impDecl.Specs, newImport)

		if lastImport >= 0 {
			insertPos = f.Decls[lastImport].End()
		} else {
			// There are no existing imports.
			// Our new import goes after the package declaration.
			insertPos = f.Name.End()
		}

		// Print the whole import declaration.
		newText = fmt.Sprintf("\n\nimport %s", printImportSpec(fset, newImport))
	} else {
		// Insert new import at insertAt.
		insertAt := 0
		if impIndex >= 0 {
			// insert after the found import
			insertAt = impIndex + 1
		}

		insertPos = impDecl.Lparen + 1 // insert after the parenthesis
		if len(impDecl.Specs) > 0 {
			insertPos = impDecl.Specs[0].Pos() // insert at the beginning
		}
		if insertAt > 0 {
			// If there is a comment after an existing import, preserve the comment
			// position by adding the new import after the comment.
			if spec, ok := impDecl.Specs[insertAt-1].(*ast.ImportSpec); ok && spec.Comment != nil {
				insertPos = spec.Comment.End()
			} else {
				// Assign same position as the previous import,
				// so that the sorter sees it as being in the same block.
				insertPos = impDecl.Specs[insertAt-1].End()
			}
		}

		// Print this import declaration.
		newText = fmt.Sprintf("\n\t%s", printImportSpec(fset, newImport))
	}

	// If we didn't find a valid insert position, return no edits.
	if !insertPos.IsValid() {
		return edits, nil
	}

	// Make sure that we are printed after any comments that start on the same line.
	file := fset.File(insertPos)
	pkgLine := file.Line(insertPos)
	for _, c := range f.Comments {
		if file.Line(c.Pos()) > pkgLine {
			break
		}
		if c.End() > insertPos {
			insertPos = c.End()
		}
	}

	rng := span.NewRange(fset, insertPos, insertPos)
	spn, err := rng.Span()
	if err != nil {
		return nil, err
	}

	edits = append(edits, diff.TextEdit{
		Span:    spn,
		NewText: newText,
	})
	return edits, nil
}

func printImportSpec(fset *token.FileSet, spec *ast.ImportSpec) string {
	var buf bytes.Buffer
	format.Node(&buf, fset, spec)
	return buf.String()
}

func isThirdParty(importPath string) bool {
	// Third party package import path usually contains "." (".com", ".org", ...)
	// This logic is taken from golang.org/x/tools/imports package.
	return strings.Contains(importPath, ".")
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

// declImports reports whether gen contains an import of path.
func declImports(gen *ast.GenDecl, path string) bool {
	if gen.Tok != token.IMPORT {
		return false
	}
	for _, spec := range gen.Specs {
		impspec := spec.(*ast.ImportSpec)
		if importPath(impspec) == path {
			return true
		}
	}
	return false
}

// matchLen returns the length of the longest path segment prefix shared by x and y.
func matchLen(x, y string) int {
	n := 0
	for i := 0; i < len(x) && i < len(y) && x[i] == y[i]; i++ {
		if x[i] == '/' {
			n++
		}
	}
	return n
}
