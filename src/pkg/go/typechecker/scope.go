// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DEPRECATED FILE - WILL GO AWAY EVENTUALLY.
//
// Scope handling is now done in go/parser.
// The functionality here is only present to
// keep the typechecker running for now.

package typechecker

import "go/ast"


func (tc *typechecker) openScope() *ast.Scope {
	tc.topScope = ast.NewScope(tc.topScope)
	return tc.topScope
}


func (tc *typechecker) closeScope() {
	tc.topScope = tc.topScope.Outer
}


// declInScope declares an object of a given kind and name in scope and sets the object's Decl and N fields.
// It returns the newly allocated object. If an object with the same name already exists in scope, an error
// is reported and the object is not inserted.
func (tc *typechecker) declInScope(scope *ast.Scope, kind ast.ObjKind, name *ast.Ident, decl interface{}, n int) *ast.Object {
	obj := ast.NewObj(kind, name.Name)
	obj.Decl = decl
	//obj.N = n
	name.Obj = obj
	if name.Name != "_" {
		if alt := scope.Insert(obj); alt != nil {
			tc.Errorf(name.Pos(), "%s already declared at %s", name.Name, tc.fset.Position(alt.Pos()).String())
		}
	}
	return obj
}


// decl is the same as declInScope(tc.topScope, ...)
func (tc *typechecker) decl(kind ast.ObjKind, name *ast.Ident, decl interface{}, n int) *ast.Object {
	return tc.declInScope(tc.topScope, kind, name, decl, n)
}


// find returns the object with the given name if visible in the current scope hierarchy.
// If no such object is found, an error is reported and a bad object is returned instead.
func (tc *typechecker) find(name *ast.Ident) (obj *ast.Object) {
	for s := tc.topScope; s != nil && obj == nil; s = s.Outer {
		obj = s.Lookup(name.Name)
	}
	if obj == nil {
		tc.Errorf(name.Pos(), "%s not declared", name.Name)
		obj = ast.NewObj(ast.Bad, name.Name)
	}
	name.Obj = obj
	return
}


// findField returns the object with the given name if visible in the type's scope.
// If no such object is found, an error is reported and a bad object is returned instead.
func (tc *typechecker) findField(typ *Type, name *ast.Ident) (obj *ast.Object) {
	// TODO(gri) This is simplistic at the moment and ignores anonymous fields.
	obj = typ.Scope.Lookup(name.Name)
	if obj == nil {
		tc.Errorf(name.Pos(), "%s not declared", name.Name)
		obj = ast.NewObj(ast.Bad, name.Name)
	}
	return
}
