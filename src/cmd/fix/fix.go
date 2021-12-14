// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/token"
	"path"
	"strconv"
)

type fix struct {
	name     string
	date     string // date that fix was introduced, in YYYY-MM-DD format
	f        func(*ast.File) bool
	desc     string
	disabled bool // whether this fix should be disabled by default
}

// main runs sort.Sort(byName(fixes)) before printing list of fixes.
type byName []fix

func (f byName) Len() int           { return len(f) }
func (f byName) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }
func (f byName) Less(i, j int) bool { return f[i].name < f[j].name }

// main runs sort.Sort(byDate(fixes)) before applying fixes.
type byDate []fix

func (f byDate) Len() int           { return len(f) }
func (f byDate) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }
func (f byDate) Less(i, j int) bool { return f[i].date < f[j].date }

var fixes []fix

func register(f fix) {
	fixes = append(fixes, f)
}

// walk traverses the AST x, calling visit(y) for each node y in the tree but
// also with a pointer to each ast.Expr, ast.Stmt, and *ast.BlockStmt,
// in a bottom-up traversal.
func walk(x interface{}, visit func(interface{})) {
	walkBeforeAfter(x, nop, visit)
}

func nop(interface{}) {}

// walkBeforeAfter is like walk but calls before(x) before traversing
// x's children and after(x) afterward.
func walkBeforeAfter(x interface{}, before, after func(interface{})) {
	before(x)

	switch n := x.(type) {
	default:
		panic(fmt.Errorf("unexpected type %T in walkBeforeAfter", x))

	case nil:

	// pointers to interfaces
	case *ast.Decl:
		walkBeforeAfter(*n, before, after)
	case *ast.Expr:
		walkBeforeAfter(*n, before, after)
	case *ast.Spec:
		walkBeforeAfter(*n, before, after)
	case *ast.Stmt:
		walkBeforeAfter(*n, before, after)

	// pointers to struct pointers
	case **ast.BlockStmt:
		walkBeforeAfter(*n, before, after)
	case **ast.CallExpr:
		walkBeforeAfter(*n, before, after)
	case **ast.FieldList:
		walkBeforeAfter(*n, before, after)
	case **ast.FuncType:
		walkBeforeAfter(*n, before, after)
	case **ast.Ident:
		walkBeforeAfter(*n, before, after)
	case **ast.BasicLit:
		walkBeforeAfter(*n, before, after)

	// pointers to slices
	case *[]ast.Decl:
		walkBeforeAfter(*n, before, after)
	case *[]ast.Expr:
		walkBeforeAfter(*n, before, after)
	case *[]*ast.File:
		walkBeforeAfter(*n, before, after)
	case *[]*ast.Ident:
		walkBeforeAfter(*n, before, after)
	case *[]ast.Spec:
		walkBeforeAfter(*n, before, after)
	case *[]ast.Stmt:
		walkBeforeAfter(*n, before, after)

	// These are ordered and grouped to match ../../go/ast/ast.go
	case *ast.Field:
		walkBeforeAfter(&n.Names, before, after)
		walkBeforeAfter(&n.Type, before, after)
		walkBeforeAfter(&n.Tag, before, after)
	case *ast.FieldList:
		for _, field := range n.List {
			walkBeforeAfter(field, before, after)
		}
	case *ast.BadExpr:
	case *ast.Ident:
	case *ast.Ellipsis:
		walkBeforeAfter(&n.Elt, before, after)
	case *ast.BasicLit:
	case *ast.FuncLit:
		walkBeforeAfter(&n.Type, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.CompositeLit:
		walkBeforeAfter(&n.Type, before, after)
		walkBeforeAfter(&n.Elts, before, after)
	case *ast.ParenExpr:
		walkBeforeAfter(&n.X, before, after)
	case *ast.SelectorExpr:
		walkBeforeAfter(&n.X, before, after)
	case *ast.IndexExpr:
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Index, before, after)
	case *ast.IndexListExpr:
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Indices, before, after)
	case *ast.SliceExpr:
		walkBeforeAfter(&n.X, before, after)
		if n.Low != nil {
			walkBeforeAfter(&n.Low, before, after)
		}
		if n.High != nil {
			walkBeforeAfter(&n.High, before, after)
		}
	case *ast.TypeAssertExpr:
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Type, before, after)
	case *ast.CallExpr:
		walkBeforeAfter(&n.Fun, before, after)
		walkBeforeAfter(&n.Args, before, after)
	case *ast.StarExpr:
		walkBeforeAfter(&n.X, before, after)
	case *ast.UnaryExpr:
		walkBeforeAfter(&n.X, before, after)
	case *ast.BinaryExpr:
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Y, before, after)
	case *ast.KeyValueExpr:
		walkBeforeAfter(&n.Key, before, after)
		walkBeforeAfter(&n.Value, before, after)

	case *ast.ArrayType:
		walkBeforeAfter(&n.Len, before, after)
		walkBeforeAfter(&n.Elt, before, after)
	case *ast.StructType:
		walkBeforeAfter(&n.Fields, before, after)
	case *ast.FuncType:
		if n.TypeParams != nil {
			walkBeforeAfter(&n.TypeParams, before, after)
		}
		walkBeforeAfter(&n.Params, before, after)
		if n.Results != nil {
			walkBeforeAfter(&n.Results, before, after)
		}
	case *ast.InterfaceType:
		walkBeforeAfter(&n.Methods, before, after)
	case *ast.MapType:
		walkBeforeAfter(&n.Key, before, after)
		walkBeforeAfter(&n.Value, before, after)
	case *ast.ChanType:
		walkBeforeAfter(&n.Value, before, after)

	case *ast.BadStmt:
	case *ast.DeclStmt:
		walkBeforeAfter(&n.Decl, before, after)
	case *ast.EmptyStmt:
	case *ast.LabeledStmt:
		walkBeforeAfter(&n.Stmt, before, after)
	case *ast.ExprStmt:
		walkBeforeAfter(&n.X, before, after)
	case *ast.SendStmt:
		walkBeforeAfter(&n.Chan, before, after)
		walkBeforeAfter(&n.Value, before, after)
	case *ast.IncDecStmt:
		walkBeforeAfter(&n.X, before, after)
	case *ast.AssignStmt:
		walkBeforeAfter(&n.Lhs, before, after)
		walkBeforeAfter(&n.Rhs, before, after)
	case *ast.GoStmt:
		walkBeforeAfter(&n.Call, before, after)
	case *ast.DeferStmt:
		walkBeforeAfter(&n.Call, before, after)
	case *ast.ReturnStmt:
		walkBeforeAfter(&n.Results, before, after)
	case *ast.BranchStmt:
	case *ast.BlockStmt:
		walkBeforeAfter(&n.List, before, after)
	case *ast.IfStmt:
		walkBeforeAfter(&n.Init, before, after)
		walkBeforeAfter(&n.Cond, before, after)
		walkBeforeAfter(&n.Body, before, after)
		walkBeforeAfter(&n.Else, before, after)
	case *ast.CaseClause:
		walkBeforeAfter(&n.List, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.SwitchStmt:
		walkBeforeAfter(&n.Init, before, after)
		walkBeforeAfter(&n.Tag, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.TypeSwitchStmt:
		walkBeforeAfter(&n.Init, before, after)
		walkBeforeAfter(&n.Assign, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.CommClause:
		walkBeforeAfter(&n.Comm, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.SelectStmt:
		walkBeforeAfter(&n.Body, before, after)
	case *ast.ForStmt:
		walkBeforeAfter(&n.Init, before, after)
		walkBeforeAfter(&n.Cond, before, after)
		walkBeforeAfter(&n.Post, before, after)
		walkBeforeAfter(&n.Body, before, after)
	case *ast.RangeStmt:
		walkBeforeAfter(&n.Key, before, after)
		walkBeforeAfter(&n.Value, before, after)
		walkBeforeAfter(&n.X, before, after)
		walkBeforeAfter(&n.Body, before, after)

	case *ast.ImportSpec:
	case *ast.ValueSpec:
		walkBeforeAfter(&n.Type, before, after)
		walkBeforeAfter(&n.Values, before, after)
		walkBeforeAfter(&n.Names, before, after)
	case *ast.TypeSpec:
		if n.TypeParams != nil {
			walkBeforeAfter(&n.TypeParams, before, after)
		}
		walkBeforeAfter(&n.Type, before, after)

	case *ast.BadDecl:
	case *ast.GenDecl:
		walkBeforeAfter(&n.Specs, before, after)
	case *ast.FuncDecl:
		if n.Recv != nil {
			walkBeforeAfter(&n.Recv, before, after)
		}
		walkBeforeAfter(&n.Type, before, after)
		if n.Body != nil {
			walkBeforeAfter(&n.Body, before, after)
		}

	case *ast.File:
		walkBeforeAfter(&n.Decls, before, after)

	case *ast.Package:
		walkBeforeAfter(&n.Files, before, after)

	case []*ast.File:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []ast.Decl:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []ast.Expr:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []*ast.Ident:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []ast.Stmt:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	case []ast.Spec:
		for i := range n {
			walkBeforeAfter(&n[i], before, after)
		}
	}
	after(x)
}

// imports reports whether f imports path.
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

// isTopName reports whether n is a top-level unresolved identifier with the given name.
func isTopName(n ast.Expr, name string) bool {
	id, ok := n.(*ast.Ident)
	return ok && id.Name == name && id.Obj == nil
}

// renameTop renames all references to the top-level name old.
// It reports whether it makes any changes.
func renameTop(f *ast.File, old, new string) bool {
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
	walk(f, func(n interface{}) {
		id, ok := n.(*ast.Ident)
		if ok && isTopName(id, old) {
			id.Name = new
			fixed = true
		}
		if ok && id.Obj != nil && id.Name == old && id.Obj.Name == new {
			id.Name = id.Obj.Name
			fixed = true
		}
	})

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

// addImport adds the import path to the file f, if absent.
func addImport(f *ast.File, ipath string) (added bool) {
	if imports(f, ipath) {
		return false
	}

	// Determine name of import.
	// Assume added imports follow convention of using last element.
	_, name := path.Split(ipath)

	// Rename any conflicting top-level references from name to name_.
	renameTop(f, name, name+"_")

	newImport := &ast.ImportSpec{
		Path: &ast.BasicLit{
			Kind:  token.STRING,
			Value: strconv.Quote(ipath),
		},
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

	f.Imports = append(f.Imports, newImport)
	return true
}

// deleteImport deletes the import path from the file f, if present.
func deleteImport(f *ast.File, path string) (deleted bool) {
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

// rewriteImport rewrites any import of path oldPath to path newPath.
func rewriteImport(f *ast.File, oldPath, newPath string) (rewrote bool) {
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
