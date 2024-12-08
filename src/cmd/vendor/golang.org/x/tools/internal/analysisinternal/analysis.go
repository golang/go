// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package analysisinternal provides gopls' internal analyses with a
// number of helper functions that operate on typed syntax trees.
package analysisinternal

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/scanner"
	"go/token"
	"go/types"
	"os"
	pathpkg "path"

	"golang.org/x/tools/go/analysis"
)

func TypeErrorEndPos(fset *token.FileSet, src []byte, start token.Pos) token.Pos {
	// Get the end position for the type error.
	file := fset.File(start)
	if file == nil {
		return start
	}
	if offset := file.PositionFor(start, false).Offset; offset > len(src) {
		return start
	} else {
		src = src[offset:]
	}

	// Attempt to find a reasonable end position for the type error.
	//
	// TODO(rfindley): the heuristic implemented here is unclear. It looks like
	// it seeks the end of the primary operand starting at start, but that is not
	// quite implemented (for example, given a func literal this heuristic will
	// return the range of the func keyword).
	//
	// We should formalize this heuristic, or deprecate it by finally proposing
	// to add end position to all type checker errors.
	//
	// Nevertheless, ensure that the end position at least spans the current
	// token at the cursor (this was golang/go#69505).
	end := start
	{
		var s scanner.Scanner
		fset := token.NewFileSet()
		f := fset.AddFile("", fset.Base(), len(src))
		s.Init(f, src, nil /* no error handler */, scanner.ScanComments)
		pos, tok, lit := s.Scan()
		if tok != token.SEMICOLON && token.Pos(f.Base()) <= pos && pos <= token.Pos(f.Base()+f.Size()) {
			off := file.Offset(pos) + len(lit)
			src = src[off:]
			end += token.Pos(off)
		}
	}

	// Look for bytes that might terminate the current operand. See note above:
	// this is imprecise.
	if width := bytes.IndexAny(src, " \n,():;[]+-*/"); width > 0 {
		end += token.Pos(width)
	}
	return end
}

// StmtToInsertVarBefore returns the ast.Stmt before which we can
// safely insert a new var declaration, or nil if the path denotes a
// node outside any statement.
//
// Basic Example:
//
//	z := 1
//	y := z + x
//
// If x is undeclared, then this function would return `y := z + x`, so that we
// can insert `x := ` on the line before `y := z + x`.
//
// If stmt example:
//
//	if z == 1 {
//	} else if z == y {}
//
// If y is undeclared, then this function would return `if z == 1 {`, because we cannot
// insert a statement between an if and an else if statement. As a result, we need to find
// the top of the if chain to insert `y := ` before.
func StmtToInsertVarBefore(path []ast.Node) ast.Stmt {
	enclosingIndex := -1
	for i, p := range path {
		if _, ok := p.(ast.Stmt); ok {
			enclosingIndex = i
			break
		}
	}
	if enclosingIndex == -1 {
		return nil // no enclosing statement: outside function
	}
	enclosingStmt := path[enclosingIndex]
	switch enclosingStmt.(type) {
	case *ast.IfStmt:
		// The enclosingStmt is inside of the if declaration,
		// We need to check if we are in an else-if stmt and
		// get the base if statement.
		// TODO(adonovan): for non-constants, it may be preferable
		// to add the decl as the Init field of the innermost
		// enclosing ast.IfStmt.
		return baseIfStmt(path, enclosingIndex)
	case *ast.CaseClause:
		// Get the enclosing switch stmt if the enclosingStmt is
		// inside of the case statement.
		for i := enclosingIndex + 1; i < len(path); i++ {
			if node, ok := path[i].(*ast.SwitchStmt); ok {
				return node
			} else if node, ok := path[i].(*ast.TypeSwitchStmt); ok {
				return node
			}
		}
	}
	if len(path) <= enclosingIndex+1 {
		return enclosingStmt.(ast.Stmt)
	}
	// Check if the enclosing statement is inside another node.
	switch expr := path[enclosingIndex+1].(type) {
	case *ast.IfStmt:
		// Get the base if statement.
		return baseIfStmt(path, enclosingIndex+1)
	case *ast.ForStmt:
		if expr.Init == enclosingStmt || expr.Post == enclosingStmt {
			return expr
		}
	case *ast.SwitchStmt, *ast.TypeSwitchStmt:
		return expr.(ast.Stmt)
	}
	return enclosingStmt.(ast.Stmt)
}

// baseIfStmt walks up the if/else-if chain until we get to
// the top of the current if chain.
func baseIfStmt(path []ast.Node, index int) ast.Stmt {
	stmt := path[index]
	for i := index + 1; i < len(path); i++ {
		if node, ok := path[i].(*ast.IfStmt); ok && node.Else == stmt {
			stmt = node
			continue
		}
		break
	}
	return stmt.(ast.Stmt)
}

// WalkASTWithParent walks the AST rooted at n. The semantics are
// similar to ast.Inspect except it does not call f(nil).
func WalkASTWithParent(n ast.Node, f func(n ast.Node, parent ast.Node) bool) {
	var ancestors []ast.Node
	ast.Inspect(n, func(n ast.Node) (recurse bool) {
		if n == nil {
			ancestors = ancestors[:len(ancestors)-1]
			return false
		}

		var parent ast.Node
		if len(ancestors) > 0 {
			parent = ancestors[len(ancestors)-1]
		}
		ancestors = append(ancestors, n)
		return f(n, parent)
	})
}

// MatchingIdents finds the names of all identifiers in 'node' that match any of the given types.
// 'pos' represents the position at which the identifiers may be inserted. 'pos' must be within
// the scope of each of identifier we select. Otherwise, we will insert a variable at 'pos' that
// is unrecognized.
func MatchingIdents(typs []types.Type, node ast.Node, pos token.Pos, info *types.Info, pkg *types.Package) map[types.Type][]string {

	// Initialize matches to contain the variable types we are searching for.
	matches := make(map[types.Type][]string)
	for _, typ := range typs {
		if typ == nil {
			continue // TODO(adonovan): is this reachable?
		}
		matches[typ] = nil // create entry
	}

	seen := map[types.Object]struct{}{}
	ast.Inspect(node, func(n ast.Node) bool {
		if n == nil {
			return false
		}
		// Prevent circular definitions. If 'pos' is within an assignment statement, do not
		// allow any identifiers in that assignment statement to be selected. Otherwise,
		// we could do the following, where 'x' satisfies the type of 'f0':
		//
		// x := fakeStruct{f0: x}
		//
		if assign, ok := n.(*ast.AssignStmt); ok && pos > assign.Pos() && pos <= assign.End() {
			return false
		}
		if n.End() > pos {
			return n.Pos() <= pos
		}
		ident, ok := n.(*ast.Ident)
		if !ok || ident.Name == "_" {
			return true
		}
		obj := info.Defs[ident]
		if obj == nil || obj.Type() == nil {
			return true
		}
		if _, ok := obj.(*types.TypeName); ok {
			return true
		}
		// Prevent duplicates in matches' values.
		if _, ok = seen[obj]; ok {
			return true
		}
		seen[obj] = struct{}{}
		// Find the scope for the given position. Then, check whether the object
		// exists within the scope.
		innerScope := pkg.Scope().Innermost(pos)
		if innerScope == nil {
			return true
		}
		_, foundObj := innerScope.LookupParent(ident.Name, pos)
		if foundObj != obj {
			return true
		}
		// The object must match one of the types that we are searching for.
		// TODO(adonovan): opt: use typeutil.Map?
		if names, ok := matches[obj.Type()]; ok {
			matches[obj.Type()] = append(names, ident.Name)
		} else {
			// If the object type does not exactly match
			// any of the target types, greedily find the first
			// target type that the object type can satisfy.
			for typ := range matches {
				if equivalentTypes(obj.Type(), typ) {
					matches[typ] = append(matches[typ], ident.Name)
				}
			}
		}
		return true
	})
	return matches
}

func equivalentTypes(want, got types.Type) bool {
	if types.Identical(want, got) {
		return true
	}
	// Code segment to help check for untyped equality from (golang/go#32146).
	if rhs, ok := want.(*types.Basic); ok && rhs.Info()&types.IsUntyped > 0 {
		if lhs, ok := got.Underlying().(*types.Basic); ok {
			return rhs.Info()&types.IsConstType == lhs.Info()&types.IsConstType
		}
	}
	return types.AssignableTo(want, got)
}

// MakeReadFile returns a simple implementation of the Pass.ReadFile function.
func MakeReadFile(pass *analysis.Pass) func(filename string) ([]byte, error) {
	return func(filename string) ([]byte, error) {
		if err := CheckReadable(pass, filename); err != nil {
			return nil, err
		}
		return os.ReadFile(filename)
	}
}

// CheckReadable enforces the access policy defined by the ReadFile field of [analysis.Pass].
func CheckReadable(pass *analysis.Pass, filename string) error {
	if slicesContains(pass.OtherFiles, filename) ||
		slicesContains(pass.IgnoredFiles, filename) {
		return nil
	}
	for _, f := range pass.Files {
		if pass.Fset.File(f.FileStart).Name() == filename {
			return nil
		}
	}
	return fmt.Errorf("Pass.ReadFile: %s is not among OtherFiles, IgnoredFiles, or names of Files", filename)
}

// TODO(adonovan): use go1.21 slices.Contains.
func slicesContains[S ~[]E, E comparable](slice S, x E) bool {
	for _, elem := range slice {
		if elem == x {
			return true
		}
	}
	return false
}

// AddImport checks whether this file already imports pkgpath and
// that import is in scope at pos. If so, it returns the name under
// which it was imported and a zero edit. Otherwise, it adds a new
// import of pkgpath, using a name derived from the preferred name,
// and returns the chosen name along with the edit for the new import.
//
// It does not mutate its arguments.
func AddImport(info *types.Info, file *ast.File, pos token.Pos, pkgpath, preferredName string) (name string, newImport []analysis.TextEdit) {
	// Find innermost enclosing lexical block.
	scope := info.Scopes[file].Innermost(pos)
	if scope == nil {
		panic("no enclosing lexical block")
	}

	// Is there an existing import of this package?
	// If so, are we in its scope? (not shadowed)
	for _, spec := range file.Imports {
		pkgname, ok := importedPkgName(info, spec)
		if ok && pkgname.Imported().Path() == pkgpath {
			if _, obj := scope.LookupParent(pkgname.Name(), pos); obj == pkgname {
				return pkgname.Name(), nil
			}
		}
	}

	// We must add a new import.
	// Ensure we have a fresh name.
	newName := preferredName
	for i := 0; ; i++ {
		if _, obj := scope.LookupParent(newName, pos); obj == nil {
			break // fresh
		}
		newName = fmt.Sprintf("%s%d", preferredName, i)
	}

	// For now, keep it real simple: create a new import
	// declaration before the first existing declaration (which
	// must exist), including its comments, and let goimports tidy it up.
	//
	// Use a renaming import whenever the preferred name is not
	// available, or the chosen name does not match the last
	// segment of its path.
	newText := fmt.Sprintf("import %q\n\n", pkgpath)
	if newName != preferredName || newName != pathpkg.Base(pkgpath) {
		newText = fmt.Sprintf("import %s %q\n\n", newName, pkgpath)
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
	return newName, []analysis.TextEdit{{
		Pos:     before.Pos(),
		End:     before.Pos(),
		NewText: []byte(newText),
	}}
}

// importedPkgName returns the PkgName object declared by an ImportSpec.
// TODO(adonovan): use go1.22's Info.PkgNameOf.
func importedPkgName(info *types.Info, imp *ast.ImportSpec) (*types.PkgName, bool) {
	var obj types.Object
	if imp.Name != nil {
		obj = info.Defs[imp.Name]
	} else {
		obj = info.Implicits[imp]
	}
	pkgname, ok := obj.(*types.PkgName)
	return pkgname, ok
}
