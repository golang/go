// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package astutil contains common utilities for working with the Go AST.
package astutil

import (
	"go/ast"
	"go/token"
	"path"
	"strconv"
	"strings"
)

// AddImport adds the import path to the file f, if absent.
func AddImport(f *ast.File, ipath string) (added bool) {
	return AddNamedImport(f, "", ipath)
}

// AddNamedImport adds the import path to the file f, if absent.
// If name is not empty, it is used to rename the import.
//
// For example, calling
//	AddNamedImport(f, "pathpkg", "path")
// adds
//	import pathpkg "path"
func AddNamedImport(f *ast.File, name, ipath string) (added bool) {
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
	var (
		bestMatch  = -1
		lastImport = -1
		impDecl    *ast.GenDecl
		impIndex   = -1
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

			// Compute longest shared prefix with imports in this block.
			for j, spec := range gen.Specs {
				impspec := spec.(*ast.ImportSpec)
				n := matchLen(importPath(impspec), ipath)
				if n > bestMatch {
					bestMatch = n
					impDecl = gen
					impIndex = j
				}
			}
		}
	}

	// If no import decl found, add one after the last import.
	if impDecl == nil {
		impDecl = &ast.GenDecl{
			Tok: token.IMPORT,
		}
		f.Decls = append(f.Decls, nil)
		copy(f.Decls[lastImport+2:], f.Decls[lastImport+1:])
		f.Decls[lastImport+1] = impDecl
	}

	// Ensure the import decl has parentheses, if needed.
	if len(impDecl.Specs) > 0 && !impDecl.Lparen.IsValid() {
		impDecl.Lparen = impDecl.Pos()
	}

	insertAt := impIndex + 1
	if insertAt == 0 {
		insertAt = len(impDecl.Specs)
	}
	impDecl.Specs = append(impDecl.Specs, nil)
	copy(impDecl.Specs[insertAt+1:], impDecl.Specs[insertAt:])
	impDecl.Specs[insertAt] = newImport
	if insertAt > 0 {
		// Assign same position as the previous import,
		// so that the sorter sees it as being in the same block.
		prev := impDecl.Specs[insertAt-1]
		newImport.Path.ValuePos = prev.Pos()
		newImport.EndPos = prev.Pos()
	}
	if len(impDecl.Specs) > 1 && impDecl.Lparen == 0 {
		// set Lparen to something not zero, so the printer prints
		// the full block rather just the first ImportSpec.
		impDecl.Lparen = 1
	}

	f.Imports = append(f.Imports, newImport)
	return true
}

// DeleteImport deletes the import path from the file f, if present.
func DeleteImport(f *ast.File, path string) (deleted bool) {
	oldImport := importSpec(f, path)

	// Find the import node that imports path, if any.
	for i, decl := range f.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.IMPORT {
			continue
		}
		for j, spec := range gen.Specs {
			impspec := spec.(*ast.ImportSpec)
			if oldImport != impspec {
				continue
			}

			// We found an import spec that imports path.
			// Delete it.
			deleted = true
			copy(gen.Specs[j:], gen.Specs[j+1:])
			gen.Specs = gen.Specs[:len(gen.Specs)-1]

			// If this was the last import spec in this decl,
			// delete the decl, too.
			if len(gen.Specs) == 0 {
				copy(f.Decls[i:], f.Decls[i+1:])
				f.Decls = f.Decls[:len(f.Decls)-1]
			} else if len(gen.Specs) == 1 {
				gen.Lparen = token.NoPos // drop parens
			}
			if j > 0 {
				// We deleted an entry but now there will be
				// a blank line-sized hole where the import was.
				// Close the hole by making the previous
				// import appear to "end" where this one did.
				gen.Specs[j-1].(*ast.ImportSpec).EndPos = impspec.End()
			}
			break
		}
	}

	// Delete it from f.Imports.
	for i, imp := range f.Imports {
		if imp == oldImport {
			copy(f.Imports[i:], f.Imports[i+1:])
			f.Imports = f.Imports[:len(f.Imports)-1]
			break
		}
	}

	return
}

// RewriteImport rewrites any import of path oldPath to path newPath.
func RewriteImport(f *ast.File, oldPath, newPath string) (rewrote bool) {
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

// RenameTop renames all references to the top-level name old.
// It returns true if it makes any changes.
func RenameTop(f *ast.File, old, new string) bool {
	var fixed bool

	// Rename any conflicting imports
	// (assuming package name is last element of path).
	for _, s := range f.Imports {
		if s.Name != nil {
			if s.Name.Name == old {
				s.Name.Name = new
				fixed = true
			}
		} else {
			_, thisName := path.Split(importPath(s))
			if thisName == old {
				s.Name = ast.NewIdent(new)
				fixed = true
			}
		}
	}

	// Rename any top-level declarations.
	for _, d := range f.Decls {
		switch d := d.(type) {
		case *ast.FuncDecl:
			if d.Recv == nil && d.Name.Name == old {
				d.Name.Name = new
				d.Name.Obj.Name = new
				fixed = true
			}
		case *ast.GenDecl:
			for _, s := range d.Specs {
				switch s := s.(type) {
				case *ast.TypeSpec:
					if s.Name.Name == old {
						s.Name.Name = new
						s.Name.Obj.Name = new
						fixed = true
					}
				case *ast.ValueSpec:
					for _, n := range s.Names {
						if n.Name == old {
							n.Name = new
							n.Obj.Name = new
							fixed = true
						}
					}
				}
			}
		}
	}

	// Rename top-level old to new, both unresolved names
	// (probably defined in another file) and names that resolve
	// to a declaration we renamed.
	ast.Walk(visitFn(func(n ast.Node) {
		id, ok := n.(*ast.Ident)
		if ok && isTopName(id, old) {
			id.Name = new
			fixed = true
		}
		if ok && id.Obj != nil && id.Name == old && id.Obj.Name == new {
			id.Name = id.Obj.Name
			fixed = true
		}
	}), f)

	return fixed
}

// matchLen returns the length of the longest prefix shared by x and y.
func matchLen(x, y string) int {
	i := 0
	for i < len(x) && i < len(y) && x[i] == y[i] {
		i++
	}
	return i
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
