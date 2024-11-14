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
	"go/token"
	"go/types"
	"os"
	pathpkg "path"
	"strconv"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/aliases"
)

func TypeErrorEndPos(fset *token.FileSet, src []byte, start token.Pos) token.Pos {
	// Get the end position for the type error.
	offset, end := fset.PositionFor(start, false).Offset, start
	if offset >= len(src) {
		return end
	}
	if width := bytes.IndexAny(src[offset:], " \n,():;[]+-*"); width > 0 {
		end = start + token.Pos(width)
	}
	return end
}

func ZeroValue(f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	// TODO(adonovan): think about generics, and also generic aliases.
	under := aliases.Unalias(typ)
	// Don't call Underlying unconditionally: although it removes
	// Named and Alias, it also removes TypeParam.
	if n, ok := under.(*types.Named); ok {
		under = n.Underlying()
	}
	switch under := under.(type) {
	case *types.Basic:
		switch {
		case under.Info()&types.IsNumeric != 0:
			return &ast.BasicLit{Kind: token.INT, Value: "0"}
		case under.Info()&types.IsBoolean != 0:
			return &ast.Ident{Name: "false"}
		case under.Info()&types.IsString != 0:
			return &ast.BasicLit{Kind: token.STRING, Value: `""`}
		default:
			panic(fmt.Sprintf("unknown basic type %v", under))
		}
	case *types.Chan, *types.Interface, *types.Map, *types.Pointer, *types.Signature, *types.Slice, *types.Array:
		return ast.NewIdent("nil")
	case *types.Struct:
		texpr := TypeExpr(f, pkg, typ) // typ because we want the name here.
		if texpr == nil {
			return nil
		}
		return &ast.CompositeLit{
			Type: texpr,
		}
	}
	return nil
}

// IsZeroValue checks whether the given expression is a 'zero value' (as determined by output of
// analysisinternal.ZeroValue)
func IsZeroValue(expr ast.Expr) bool {
	switch e := expr.(type) {
	case *ast.BasicLit:
		return e.Value == "0" || e.Value == `""`
	case *ast.Ident:
		return e.Name == "nil" || e.Name == "false"
	default:
		return false
	}
}

// TypeExpr returns syntax for the specified type. References to
// named types from packages other than pkg are qualified by an appropriate
// package name, as defined by the import environment of file.
func TypeExpr(f *ast.File, pkg *types.Package, typ types.Type) ast.Expr {
	switch t := typ.(type) {
	case *types.Basic:
		switch t.Kind() {
		case types.UnsafePointer:
			return &ast.SelectorExpr{X: ast.NewIdent("unsafe"), Sel: ast.NewIdent("Pointer")}
		default:
			return ast.NewIdent(t.Name())
		}
	case *types.Pointer:
		x := TypeExpr(f, pkg, t.Elem())
		if x == nil {
			return nil
		}
		return &ast.UnaryExpr{
			Op: token.MUL,
			X:  x,
		}
	case *types.Array:
		elt := TypeExpr(f, pkg, t.Elem())
		if elt == nil {
			return nil
		}
		return &ast.ArrayType{
			Len: &ast.BasicLit{
				Kind:  token.INT,
				Value: fmt.Sprintf("%d", t.Len()),
			},
			Elt: elt,
		}
	case *types.Slice:
		elt := TypeExpr(f, pkg, t.Elem())
		if elt == nil {
			return nil
		}
		return &ast.ArrayType{
			Elt: elt,
		}
	case *types.Map:
		key := TypeExpr(f, pkg, t.Key())
		value := TypeExpr(f, pkg, t.Elem())
		if key == nil || value == nil {
			return nil
		}
		return &ast.MapType{
			Key:   key,
			Value: value,
		}
	case *types.Chan:
		dir := ast.ChanDir(t.Dir())
		if t.Dir() == types.SendRecv {
			dir = ast.SEND | ast.RECV
		}
		value := TypeExpr(f, pkg, t.Elem())
		if value == nil {
			return nil
		}
		return &ast.ChanType{
			Dir:   dir,
			Value: value,
		}
	case *types.Signature:
		var params []*ast.Field
		for i := 0; i < t.Params().Len(); i++ {
			p := TypeExpr(f, pkg, t.Params().At(i).Type())
			if p == nil {
				return nil
			}
			params = append(params, &ast.Field{
				Type: p,
				Names: []*ast.Ident{
					{
						Name: t.Params().At(i).Name(),
					},
				},
			})
		}
		if t.Variadic() {
			last := params[len(params)-1]
			last.Type = &ast.Ellipsis{Elt: last.Type.(*ast.ArrayType).Elt}
		}
		var returns []*ast.Field
		for i := 0; i < t.Results().Len(); i++ {
			r := TypeExpr(f, pkg, t.Results().At(i).Type())
			if r == nil {
				return nil
			}
			returns = append(returns, &ast.Field{
				Type: r,
			})
		}
		return &ast.FuncType{
			Params: &ast.FieldList{
				List: params,
			},
			Results: &ast.FieldList{
				List: returns,
			},
		}
	case interface{ Obj() *types.TypeName }: // *types.{Alias,Named,TypeParam}
		if t.Obj().Pkg() == nil {
			return ast.NewIdent(t.Obj().Name())
		}
		if t.Obj().Pkg() == pkg {
			return ast.NewIdent(t.Obj().Name())
		}
		pkgName := t.Obj().Pkg().Name()

		// If the file already imports the package under another name, use that.
		for _, cand := range f.Imports {
			if path, _ := strconv.Unquote(cand.Path.Value); path == t.Obj().Pkg().Path() {
				if cand.Name != nil && cand.Name.Name != "" {
					pkgName = cand.Name.Name
				}
			}
		}
		if pkgName == "." {
			return ast.NewIdent(t.Obj().Name())
		}
		return &ast.SelectorExpr{
			X:   ast.NewIdent(pkgName),
			Sel: ast.NewIdent(t.Obj().Name()),
		}
	case *types.Struct:
		return ast.NewIdent(t.String())
	case *types.Interface:
		return ast.NewIdent(t.String())
	default:
		return nil
	}
}

// StmtToInsertVarBefore returns the ast.Stmt before which we can safely insert a new variable.
// Some examples:
//
// Basic Example:
// z := 1
// y := z + x
// If x is undeclared, then this function would return `y := z + x`, so that we
// can insert `x := ` on the line before `y := z + x`.
//
// If stmt example:
// if z == 1 {
// } else if z == y {}
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
		return nil
	}
	enclosingStmt := path[enclosingIndex]
	switch enclosingStmt.(type) {
	case *ast.IfStmt:
		// The enclosingStmt is inside of the if declaration,
		// We need to check if we are in an else-if stmt and
		// get the base if statement.
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
	ast.Inspect(node, func { n ->
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
		// TODO(adonovan): use go1.20 f.FileStart
		if pass.Fset.File(f.Pos()).Name() == filename {
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
func AddImport(info *types.Info, file *ast.File, pos token.Pos, pkgpath, preferredName string) (name string, newImport analysis.TextEdit) {
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
				return pkgname.Name(), analysis.TextEdit{}
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
	return newName, analysis.TextEdit{
		Pos:     before.Pos(),
		End:     before.Pos(),
		NewText: []byte(newText),
	}
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
