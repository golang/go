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
// The result is (nil, nil, false) if not found.
//
func (prog *Program) PathEnclosingInterval(imp *importer.Importer, start, end token.Pos) (pkg *Package, path []ast.Node, exact bool) {
	for path, info := range imp.Packages {
		pkg := prog.Packages[path]
		for _, f := range info.Files {
			if !tokenFileContainsPos(prog.Files.File(f.Package), start) {
				continue
			}
			if path, exact := PathEnclosingInterval(f, start, end); path != nil {
				return pkg, path, exact
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

// findEnclosingPackageLevelFunction returns the *ssa.Function
// corresponding to the package-level function enclosing path.
//
func findEnclosingPackageLevelFunction(pkg *Package, path []ast.Node) *Function {
	if n := len(path); n >= 2 { // [... {Gen,Func}Decl File]
		switch decl := path[n-2].(type) {
		case *ast.GenDecl:
			if decl.Tok == token.VAR && n >= 3 {
				// Package-level 'var' initializer.
				return pkg.Init
			}

		case *ast.FuncDecl:
			if decl.Recv == nil && decl.Name.Name == "init" {
				// Explicit init() function.
				return pkg.Init
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
			for _, meth := range mem.PtrMethods {
				if meth.Pos() == pos {
					return meth
				}
			}
		}
	}
	return nil
}
