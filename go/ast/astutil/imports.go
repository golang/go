// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package astutil contains common utilities for working with the Go AST.
package astutil // import "golang.org/x/tools/go/ast/astutil"

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"strings"
)

// AddImport adds the import path to the file f, if absent.
func AddImport(fset *token.FileSet, f *ast.File, ipath string) (added bool) {
	return AddNamedImport(fset, f, "", ipath)
}

// AddNamedImport adds the import path to the file f, if absent.
// If name is not empty, it is used to rename the import.
//
// For example, calling
//	AddNamedImport(fset, f, "pathpkg", "path")
// adds
//	import pathpkg "path"
func AddNamedImport(fset *token.FileSet, f *ast.File, name, ipath string) (added bool) {
	if imports(f, ipath) {
		return false
	}

	newImport := &ast.ImportSpec{
		Path: &ast.BasicLit{
			Kind:  token.STRING,
			Value: strconv.Quote(ipath),
		},
	}
	if name != "" {
		newImport.Name = &ast.Ident{Name: name}
	}

	// Find an import decl to add to.
	// The goal is to find an existing import
	// whose import path has the longest shared
	// prefix with ipath.
	var (
		bestMatch  = -1         // length of longest shared prefix
		lastImport = -1         // index in f.Decls of the file's final import decl
		impDecl    *ast.GenDecl // import decl containing the best match
		impIndex   = -1         // spec index in impDecl containing the best match

		isThirdPartyPath = isThirdParty(ipath)
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
				n := matchLen(p, ipath)
				if n > bestMatch || (bestMatch == 0 && !seenAnyThirdParty && isThirdPartyPath) {
					bestMatch = n
					impDecl = gen
					impIndex = j
				}
				seenAnyThirdParty = seenAnyThirdParty || isThirdParty(p)
			}
		}
	}

	// If no import decl found, add one after the last import.
	if impDecl == nil {
		impDecl = &ast.GenDecl{
			Tok: token.IMPORT,
		}
		if lastImport >= 0 {
			impDecl.TokPos = f.Decls[lastImport].End()
		} else {
			// There are no existing imports.
			// Our new import, preceded by a blank line,  goes after the package declaration
			// and after the comment, if any, that starts on the same line as the
			// package declaration.
			impDecl.TokPos = f.Package

			file := fset.File(f.Package)
			pkgLine := file.Line(f.Package)
			for _, c := range f.Comments {
				if file.Line(c.Pos()) > pkgLine {
					break
				}
				// +2 for a blank line
				impDecl.TokPos = c.End() + 2
			}
		}
		f.Decls = append(f.Decls, nil)
		copy(f.Decls[lastImport+2:], f.Decls[lastImport+1:])
		f.Decls[lastImport+1] = impDecl
	}

	// Insert new import at insertAt.
	insertAt := 0
	if impIndex >= 0 {
		// insert after the found import
		insertAt = impIndex + 1
	}
	impDecl.Specs = append(impDecl.Specs, nil)
	copy(impDecl.Specs[insertAt+1:], impDecl.Specs[insertAt:])
	impDecl.Specs[insertAt] = newImport
	pos := impDecl.Pos()
	if insertAt > 0 {
		// If there is a comment after an existing import, preserve the comment
		// position by adding the new import after the comment.
		if spec, ok := impDecl.Specs[insertAt-1].(*ast.ImportSpec); ok && spec.Comment != nil {
			pos = spec.Comment.End()
		} else {
			// Assign same position as the previous import,
			// so that the sorter sees it as being in the same block.
			pos = impDecl.Specs[insertAt-1].Pos()
		}
	}
	if newImport.Name != nil {
		newImport.Name.NamePos = pos
	}
	newImport.Path.ValuePos = pos
	newImport.EndPos = pos

	// Clean up parens. impDecl contains at least one spec.
	if len(impDecl.Specs) == 1 {
		// Remove unneeded parens.
		impDecl.Lparen = token.NoPos
	} else if !impDecl.Lparen.IsValid() {
		// impDecl needs parens added.
		impDecl.Lparen = impDecl.Specs[0].Pos()
	}

	f.Imports = append(f.Imports, newImport)

	if len(f.Decls) <= 1 {
		return true
	}

	// Merge all the import declarations into the first one.
	var first *ast.GenDecl
	for i := 0; i < len(f.Decls); i++ {
		decl := f.Decls[i]
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.IMPORT || declImports(gen, "C") {
			continue
		}
		if first == nil {
			first = gen
			continue // Don't touch the first one.
		}
		// We now know there is more than one package in this import
		// declaration. Ensure that it ends up parenthesized.
		first.Lparen = first.Pos()
		// Move the imports of the other import declaration to the first one.
		for _, spec := range gen.Specs {
			spec.(*ast.ImportSpec).Path.ValuePos = first.Pos()
			first.Specs = append(first.Specs, spec)
		}
		f.Decls = append(f.Decls[:i], f.Decls[i+1:]...)
		i--
	}

	return true
}

func isThirdParty(importPath string) bool {
	// Third party package import path usually contains "." (".com", ".org", ...)
	// This logic is taken from golang.org/x/tools/imports package.
	return strings.Contains(importPath, ".")
}

// DeleteImport deletes the import path from the file f, if present.
func DeleteImport(fset *token.FileSet, f *ast.File, path string) (deleted bool) {
	return DeleteNamedImport(fset, f, "", path)
}

// DeleteNamedImport deletes the import with the given name and path from the file f, if present.
func DeleteNamedImport(fset *token.FileSet, f *ast.File, name, path string) (deleted bool) {
	var delspecs []*ast.ImportSpec
	var delcomments []*ast.CommentGroup

	// Find the import nodes that import path, if any.
	for i := 0; i < len(f.Decls); i++ {
		decl := f.Decls[i]
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.IMPORT {
			continue
		}
		for j := 0; j < len(gen.Specs); j++ {
			spec := gen.Specs[j]
			impspec := spec.(*ast.ImportSpec)
			if impspec.Name == nil && name != "" {
				continue
			}
			if impspec.Name != nil && impspec.Name.Name != name {
				continue
			}
			if importPath(impspec) != path {
				continue
			}

			// We found an import spec that imports path.
			// Delete it.
			delspecs = append(delspecs, impspec)
			deleted = true
			copy(gen.Specs[j:], gen.Specs[j+1:])
			gen.Specs = gen.Specs[:len(gen.Specs)-1]

			// If this was the last import spec in this decl,
			// delete the decl, too.
			if len(gen.Specs) == 0 {
				copy(f.Decls[i:], f.Decls[i+1:])
				f.Decls = f.Decls[:len(f.Decls)-1]
				i--
				break
			} else if len(gen.Specs) == 1 {
				if impspec.Doc != nil {
					delcomments = append(delcomments, impspec.Doc)
				}
				if impspec.Comment != nil {
					delcomments = append(delcomments, impspec.Comment)
				}
				for _, cg := range f.Comments {
					// Found comment on the same line as the import spec.
					if cg.End() < impspec.Pos() && fset.Position(cg.End()).Line == fset.Position(impspec.Pos()).Line {
						delcomments = append(delcomments, cg)
						break
					}
				}

				spec := gen.Specs[0].(*ast.ImportSpec)

				// Move the documentation right after the import decl.
				if spec.Doc != nil {
					for fset.Position(gen.TokPos).Line+1 < fset.Position(spec.Doc.Pos()).Line {
						fset.File(gen.TokPos).MergeLine(fset.Position(gen.TokPos).Line)
					}
				}
				for _, cg := range f.Comments {
					if cg.End() < spec.Pos() && fset.Position(cg.End()).Line == fset.Position(spec.Pos()).Line {
						for fset.Position(gen.TokPos).Line+1 < fset.Position(spec.Pos()).Line {
							fset.File(gen.TokPos).MergeLine(fset.Position(gen.TokPos).Line)
						}
						break
					}
				}
			}
			if j > 0 {
				lastImpspec := gen.Specs[j-1].(*ast.ImportSpec)
				lastLine := fset.Position(lastImpspec.Path.ValuePos).Line
				line := fset.Position(impspec.Path.ValuePos).Line

				// We deleted an entry but now there may be
				// a blank line-sized hole where the import was.
				if line-lastLine > 1 {
					// There was a blank line immediately preceding the deleted import,
					// so there's no need to close the hole.
					// Do nothing.
				} else if line != fset.File(gen.Rparen).LineCount() {
					// There was no blank line. Close the hole.
					fset.File(gen.Rparen).MergeLine(line)
				}
			}
			j--
		}
	}

	// Delete imports from f.Imports.
	for i := 0; i < len(f.Imports); i++ {
		imp := f.Imports[i]
		for j, del := range delspecs {
			if imp == del {
				copy(f.Imports[i:], f.Imports[i+1:])
				f.Imports = f.Imports[:len(f.Imports)-1]
				copy(delspecs[j:], delspecs[j+1:])
				delspecs = delspecs[:len(delspecs)-1]
				i--
				break
			}
		}
	}

	// Delete comments from f.Comments.
	for i := 0; i < len(f.Comments); i++ {
		cg := f.Comments[i]
		for j, del := range delcomments {
			if cg == del {
				copy(f.Comments[i:], f.Comments[i+1:])
				f.Comments = f.Comments[:len(f.Comments)-1]
				copy(delcomments[j:], delcomments[j+1:])
				delcomments = delcomments[:len(delcomments)-1]
				i--
				break
			}
		}
	}

	if len(delspecs) > 0 {
		panic(fmt.Sprintf("deleted specs from Decls but not Imports: %v", delspecs))
	}

	return
}

// RewriteImport rewrites any import of path oldPath to path newPath.
func RewriteImport(fset *token.FileSet, f *ast.File, oldPath, newPath string) (rewrote bool) {
	for _, imp := range f.Imports {
		if importPath(imp) == oldPath {
			rewrote = true
			// record old End, because the default is to compute
			// it using the length of imp.Path.Value.
			imp.EndPos = imp.End()
			imp.Path.Value = strconv.Quote(newPath)
		}
	}
	return
}

// UsesImport reports whether a given import is used.
func UsesImport(f *ast.File, path string) (used bool) {
	spec := importSpec(f, path)
	if spec == nil {
		return
	}

	name := spec.Name.String()
	switch name {
	case "<nil>":
		// If the package name is not explicitly specified,
		// make an educated guess. This is not guaranteed to be correct.
		lastSlash := strings.LastIndex(path, "/")
		if lastSlash == -1 {
			name = path
		} else {
			name = path[lastSlash+1:]
		}
	case "_", ".":
		// Not sure if this import is used - err on the side of caution.
		return true
	}

	ast.Walk(visitFn(func(n ast.Node) {
		sel, ok := n.(*ast.SelectorExpr)
		if ok && isTopName(sel.X, name) {
			used = true
		}
	}), f)

	return
}

type visitFn func(node ast.Node)

func (fn visitFn) Visit(node ast.Node) ast.Visitor {
	fn(node)
	return fn
}

// imports returns true if f imports path.
func imports(f *ast.File, path string) bool {
	return importSpec(f, path) != nil
}

// importSpec returns the import spec if f imports path,
// or nil otherwise.
func importSpec(f *ast.File, path string) *ast.ImportSpec {
	for _, s := range f.Imports {
		if importPath(s) == path {
			return s
		}
	}
	return nil
}

// importPath returns the unquoted import path of s,
// or "" if the path is not properly quoted.
func importPath(s *ast.ImportSpec) string {
	t, err := strconv.Unquote(s.Path.Value)
	if err == nil {
		return t
	}
	return ""
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

// isTopName returns true if n is a top-level unresolved identifier with the given name.
func isTopName(n ast.Expr, name string) bool {
	id, ok := n.(*ast.Ident)
	return ok && id.Name == name && id.Obj == nil
}

// Imports returns the file imports grouped by paragraph.
func Imports(fset *token.FileSet, f *ast.File) [][]*ast.ImportSpec {
	var groups [][]*ast.ImportSpec

	for _, decl := range f.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok || genDecl.Tok != token.IMPORT {
			break
		}

		group := []*ast.ImportSpec{}

		var lastLine int
		for _, spec := range genDecl.Specs {
			importSpec := spec.(*ast.ImportSpec)
			pos := importSpec.Path.ValuePos
			line := fset.Position(pos).Line
			if lastLine > 0 && pos > 0 && line-lastLine > 1 {
				groups = append(groups, group)
				group = []*ast.ImportSpec{}
			}
			group = append(group, importSpec)
			lastLine = line
		}
		groups = append(groups, group)
	}

	return groups
}
