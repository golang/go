// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines utilities for working with source positions
// or source-level named entities ("objects").

// TODO(adonovan): test that {Value,Instruction}.Pos() positions match
// the originating syntax, as specified.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// EnclosingFunction returns the function that contains the syntax
// node denoted by path.
//
// Syntax associated with package-level variable specifications is
// enclosed by the package's init() function.
//
// Returns nil if not found; reasons might include:
//   - the node is not enclosed by any function.
//   - the node is within an anonymous function (FuncLit) and
//     its SSA function has not been created yet
//     (pkg.Build() has not yet been called).
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
func HasEnclosingFunction(pkg *Package, path []ast.Node) bool {
	return findEnclosingPackageLevelFunction(pkg, path) != nil
}

// findEnclosingPackageLevelFunction returns the Function
// corresponding to the package-level function enclosing path.
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
				for _, b := range pkg.init.Blocks {
					for _, instr := range b.Instrs {
						if instr, ok := instr.(*Call); ok {
							if callee, ok := instr.Call.Value.(*Function); ok && callee.Pkg == pkg && callee.Pos() == decl.Name.NamePos {
								return callee
							}
						}
					}
				}
				// Hack: return non-nil when SSA is not yet
				// built so that HasEnclosingFunction works.
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
			mset := pkg.Prog.MethodSets.MethodSet(types.NewPointer(mem.Type()))
			for i, n := 0, mset.Len(); i < n; i++ {
				// Don't call Program.Method: avoid creating wrappers.
				obj := mset.At(i).Obj().(*types.Func)
				if obj.Pos() == pos {
					// obj from MethodSet may not be the origin type.
					m := typeparams.OriginMethod(obj)
					return pkg.objects[m].(*Function)
				}
			}
		}
	}
	return nil
}

// goversionOf returns the goversion of a node in the package
// where the node is either a function declaration or the initial
// value of a package level variable declaration.
func goversionOf(p *Package, file *ast.File) string {
	if p.info == nil {
		return ""
	}

	// TODO(taking): Update to the following when internal/versions available:
	//	return versions.Lang(versions.FileVersions(p.info, file))
	return fileVersions(file)
}

// TODO(taking): Remove when internal/versions is available.
var fileVersions = func(file *ast.File) string { return "" }

// parses a goXX.YY version or returns a negative version on an error.
// TODO(taking): Switch to a permanent solution when internal/versions is submitted.
func parseGoVersion(x string) (major, minor int) {
	if _, err := fmt.Sscanf(x, "go%d.%d", &major, &minor); err != nil || major < 0 || minor < 0 {
		return -1, -1
	}
	return
}

// ValueForExpr returns the SSA Value that corresponds to non-constant
// expression e.
//
// It returns nil if no value was found, e.g.
//   - the expression is not lexically contained within f;
//   - f was not built with debug information; or
//   - e is a constant expression.  (For efficiency, no debug
//     information is stored for constants. Use
//     go/types.Info.Types[e].Value instead.)
//   - e is a reference to nil or a built-in function.
//   - the value was optimised away.
//
// If e is an addressable expression used in an lvalue context,
// value is the address denoted by e, and isAddr is true.
//
// The types of e (or &e, if isAddr) and the result are equal
// (modulo "untyped" bools resulting from comparisons).
//
// (Tip: to find the ssa.Value given a source position, use
// astutil.PathEnclosingInterval to locate the ast.Node, then
// EnclosingFunction to locate the Function, then ValueForExpr to find
// the ssa.Value.)
func (f *Function) ValueForExpr(e ast.Expr) (value Value, isAddr bool) {
	if f.debugInfo() { // (opt)
		e = unparen(e)
		for _, b := range f.Blocks {
			for _, instr := range b.Instrs {
				if ref, ok := instr.(*DebugRef); ok {
					if ref.Expr == e {
						return ref.X, ref.IsAddr
					}
				}
			}
		}
	}
	return
}

// --- Lookup functions for source-level named entities (types.Objects) ---

// Package returns the SSA Package corresponding to the specified
// type-checker package object.
// It returns nil if no such SSA package has been created.
func (prog *Program) Package(obj *types.Package) *Package {
	return prog.packages[obj]
}

// packageLevelMember returns the package-level member corresponding to
// the specified named object, which may be a package-level const
// (*NamedConst), var (*Global) or func (*Function) of some package in
// prog.  It returns nil if the object is not found.
func (prog *Program) packageLevelMember(obj types.Object) Member {
	if pkg, ok := prog.packages[obj.Pkg()]; ok {
		return pkg.objects[obj]
	}
	return nil
}

// originFunc returns the package-level generic function that is the
// origin of obj. If returns nil if the generic function is not found.
func (prog *Program) originFunc(obj *types.Func) *Function {
	return prog.declaredFunc(typeparams.OriginMethod(obj))
}

// FuncValue returns the concrete Function denoted by the source-level
// named function obj, or nil if obj denotes an interface method.
//
// TODO(adonovan): check the invariant that obj.Type() matches the
// result's Signature, both in the params/results and in the receiver.
func (prog *Program) FuncValue(obj *types.Func) *Function {
	fn, _ := prog.packageLevelMember(obj).(*Function)
	return fn
}

// ConstValue returns the SSA Value denoted by the source-level named
// constant obj.
func (prog *Program) ConstValue(obj *types.Const) *Const {
	// TODO(adonovan): opt: share (don't reallocate)
	// Consts for const objects and constant ast.Exprs.

	// Universal constant? {true,false,nil}
	if obj.Parent() == types.Universe {
		return NewConst(obj.Val(), obj.Type())
	}
	// Package-level named constant?
	if v := prog.packageLevelMember(obj); v != nil {
		return v.(*NamedConst).Value
	}
	return NewConst(obj.Val(), obj.Type())
}

// VarValue returns the SSA Value that corresponds to a specific
// identifier denoting the source-level named variable obj.
//
// VarValue returns nil if a local variable was not found, perhaps
// because its package was not built, the debug information was not
// requested during SSA construction, or the value was optimized away.
//
// ref is the path to an ast.Ident (e.g. from PathEnclosingInterval),
// and that ident must resolve to obj.
//
// pkg is the package enclosing the reference.  (A reference to a var
// always occurs within a function, so we need to know where to find it.)
//
// If the identifier is a field selector and its base expression is
// non-addressable, then VarValue returns the value of that field.
// For example:
//
//	func f() struct {x int}
//	f().x  // VarValue(x) returns a *Field instruction of type int
//
// All other identifiers denote addressable locations (variables).
// For them, VarValue may return either the variable's address or its
// value, even when the expression is evaluated only for its value; the
// situation is reported by isAddr, the second component of the result.
//
// If !isAddr, the returned value is the one associated with the
// specific identifier.  For example,
//
//	var x int    // VarValue(x) returns Const 0 here
//	x = 1        // VarValue(x) returns Const 1 here
//
// It is not specified whether the value or the address is returned in
// any particular case, as it may depend upon optimizations performed
// during SSA code generation, such as registerization, constant
// folding, avoidance of materialization of subexpressions, etc.
func (prog *Program) VarValue(obj *types.Var, pkg *Package, ref []ast.Node) (value Value, isAddr bool) {
	// All references to a var are local to some function, possibly init.
	fn := EnclosingFunction(pkg, ref)
	if fn == nil {
		return // e.g. def of struct field; SSA not built?
	}

	id := ref[0].(*ast.Ident)

	// Defining ident of a parameter?
	if id.Pos() == obj.Pos() {
		for _, param := range fn.Params {
			if param.Object() == obj {
				return param, false
			}
		}
	}

	// Other ident?
	for _, b := range fn.Blocks {
		for _, instr := range b.Instrs {
			if dr, ok := instr.(*DebugRef); ok {
				if dr.Pos() == id.Pos() {
					return dr.X, dr.IsAddr
				}
			}
		}
	}

	// Defining ident of package-level var?
	if v := prog.packageLevelMember(obj); v != nil {
		return v.(*Global), true
	}

	return // e.g. debug info not requested, or var optimized away
}
