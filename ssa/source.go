package ssa

// This file defines utilities for working with source positions.

// TODO(adonovan): move this and source_ast.go to a new subpackage
// since neither depends on SSA internals.

import (
	"code.google.com/p/go.tools/importer"
	"go/ast"
	"go/token"
)

// TODO(adonovan): make this a method: func (*token.File) Contains(token.Pos)
func tokenFileContainsPos(f *token.File, pos token.Pos) bool {
	p := int(pos)
	base := f.Base()
	return base <= p && p < base+f.Size()
}

// PathEnclosingInterval returns the Package and ast.Node that
// contain source interval [start, end), and all the node's ancestors
// up to the AST root.  It searches all files of all packages in the
// program prog.  exact is defined as for standalone
// PathEnclosingInterval.
//
// imp provides ASTs for the program's packages.
//
// pkg may be nil if no SSA package has yet been created for the found
// package.  Call prog.CreatePackages(imp) to avoid this.
//
// The result is (nil, nil, false) if not found.
//
func (prog *Program) PathEnclosingInterval(imp *importer.Importer, start, end token.Pos) (pkg *Package, path []ast.Node, exact bool) {
	for importPath, info := range imp.Packages {
		for _, f := range info.Files {
			if !tokenFileContainsPos(prog.Fset.File(f.Package), start) {
				continue
			}
			if path, exact := PathEnclosingInterval(f, start, end); path != nil {
				return prog.PackagesByPath[importPath], path, exact
			}
		}
	}
	return nil, nil, false
}

// EnclosingFunction returns the function that contains the syntax
// node denoted by path.
//
// Syntax associated with package-level variable specifications is
// enclosed by the package's init() function.
//
// Returns nil if not found; reasons might include:
//    - the node is not enclosed by any function.
//    - the node is within an anonymous function (FuncLit) and
//      its SSA function has not been created yet (pkg.BuildPackage()
//      has not yet been called).
//
func EnclosingFunction(pkg *Package, path []ast.Node) *Function {
	// Start with package-level function...
	fn := findEnclosingPackageLevelFunction(pkg, path)
	if fn == nil {
		return nil // not in any function
	}

	// ...then walk down the nested anonymous functions.
	n := len(path)
outer:
	for i := range path {
		if lit, ok := path[n-1-i].(*ast.FuncLit); ok {
			for _, anon := range fn.AnonFuncs {
				if anon.Pos() == lit.Type.Func {
					fn = anon
					continue outer
				}
			}
			// SSA function not found:
			// - package not yet built, or maybe
			// - builder skipped FuncLit in dead block
			//   (in principle; but currently the Builder
			//   generates even dead FuncLits).
			return nil
		}
	}
	return fn
}

// HasEnclosingFunction returns true if the AST node denoted by path
// is contained within the declaration of some function or
// package-level variable.
//
// Unlike EnclosingFunction, the behaviour of this function does not
// depend on whether SSA code for pkg has been built, so it can be
// used to quickly reject check inputs that will cause
// EnclosingFunction to fail, prior to SSA building.
//
func HasEnclosingFunction(pkg *Package, path []ast.Node) bool {
	return findEnclosingPackageLevelFunction(pkg, path) != nil
}

// findEnclosingPackageLevelFunction returns the Function
// corresponding to the package-level function enclosing path.
//
func findEnclosingPackageLevelFunction(pkg *Package, path []ast.Node) *Function {
	if n := len(path); n >= 2 { // [... {Gen,Func}Decl File]
		switch decl := path[n-2].(type) {
		case *ast.GenDecl:
			if decl.Tok == token.VAR && n >= 3 {
				// Package-level 'var' initializer.
				return pkg.init
			}

		case *ast.FuncDecl:
			if decl.Recv == nil && decl.Name.Name == "init" {
				// Explicit init() function.
				return pkg.init
			}
			// Declared function/method.
			return findNamedFunc(pkg, decl.Name.NamePos)
		}
	}
	return nil // not in any function
}

// findNamedFunc returns the named function whose FuncDecl.Ident is at
// position pos.
//
func findNamedFunc(pkg *Package, pos token.Pos) *Function {
	// Look at all package members and method sets of named types.
	// Not very efficient.
	for _, mem := range pkg.Members {
		switch mem := mem.(type) {
		case *Function:
			if mem.Pos() == pos {
				return mem
			}
		case *Type:
			for _, meth := range pkg.Prog.MethodSet(mem.Type()) {
				if meth.Synthetic == "" && meth.Pos() == pos {
					return meth
				}
			}
			for _, meth := range pkg.Prog.MethodSet(pointer(mem.Type())) {
				if meth.Synthetic == "" && meth.Pos() == pos {
					return meth
				}
			}
		}
	}
	return nil
}

// CanonicalPos returns the canonical position of the AST node n,
//
// For each Node kind that may generate an SSA Value or Instruction,
// exactly one token within it is designated as "canonical".  The
// position of that token is returned by {Value,Instruction}.Pos().
// The specifications of those methods determine the implementation of
// this function.
//
// TODO(adonovan): test coverage.
//
func CanonicalPos(n ast.Node) token.Pos {
	// Comments show the Value/Instruction kinds v that may be
	// created by n such that CanonicalPos(n) == v.Pos().
	switch n := n.(type) {
	case *ast.ParenExpr:
		return CanonicalPos(n.X)

	case *ast.CallExpr:
		// f(x):    *Call, *Go, *Defer, *Literal (e.g. len)
		// T(x):    *ChangeType, *Convert, *MakeInterface, *ChangeInterface, *Literal.
		// make():  *MakeMap, *MakeChan, *MakeSlice.
		// new():   *Alloc.
		// panic(): *Panic.
		return n.Lparen

	case *ast.BasicLit:
		return n.ValuePos // *Literal

	case *ast.Ident:
		return n.NamePos // *Parameter, *Alloc, *Capture, *Literal

	case *ast.TypeAssertExpr:
		return n.Lparen // *ChangeInterface or *TypeAssertExpr

	case *ast.SelectorExpr:
		return n.Sel.NamePos // *MakeClosure, *Field, *FieldAddr, *Literal

	case *ast.FuncLit:
		return n.Type.Func // *Function or *MakeClosure

	case *ast.CompositeLit:
		return n.Lbrace // *Alloc or *Slice

	case *ast.BinaryExpr:
		return n.OpPos // *Phi, *BinOp or *Literal

	case *ast.UnaryExpr:
		return n.OpPos // *Phi, *UnOp, or *Literal

	case *ast.IndexExpr:
		return n.Lbrack // *Index or *IndexAddr

	case *ast.SliceExpr:
		return n.Lbrack // *Slice

	case *ast.SelectStmt:
		return n.Select // *Select

	case *ast.RangeStmt:
		return n.For // *Range

	case *ast.ReturnStmt:
		return n.Return // *Ret

	case *ast.SendStmt:
		return n.Arrow // *Send

	case *ast.StarExpr:
		return n.Star // *Store

	case *ast.KeyValueExpr:
		return n.Colon // *MapUpdate
	}

	return token.NoPos
}
