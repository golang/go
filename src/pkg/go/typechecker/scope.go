// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements scope support functions.

package typechecker

import (
	"fmt"
	"go/ast"
	"go/token"
)


func (tc *typechecker) openScope() *ast.Scope {
	tc.topScope = ast.NewScope(tc.topScope)
	return tc.topScope
}


func (tc *typechecker) closeScope() {
	tc.topScope = tc.topScope.Outer
}


// objPos computes the source position of the declaration of an object name.
// Only required for error reporting, so doesn't have to be fast.
func objPos(obj *ast.Object) (pos token.Pos) {
	switch d := obj.Decl.(type) {
	case *ast.Field:
		for _, n := range d.Names {
			if n.Name == obj.Name {
				return n.Pos()
			}
		}
	case *ast.ValueSpec:
		for _, n := range d.Names {
			if n.Name == obj.Name {
				return n.Pos()
			}
		}
	case *ast.TypeSpec:
		return d.Name.Pos()
	case *ast.FuncDecl:
		return d.Name.Pos()
	}
	if debug {
		fmt.Printf("decl = %T\n", obj.Decl)
	}
	panic("unreachable")
}


// declInScope declares an object of a given kind and name in scope and sets the object's Decl and N fields.
// It returns the newly allocated object. If an object with the same name already exists in scope, an error
// is reported and the object is not inserted.
// (Objects with _ name are always inserted into a scope without errors, but they cannot be found.)
func (tc *typechecker) declInScope(scope *ast.Scope, kind ast.Kind, name *ast.Ident, decl interface{}, n int) *ast.Object {
	obj := ast.NewObj(kind, name.Name)
	obj.Decl = decl
	obj.N = n
	name.Obj = obj
	if alt := scope.Insert(obj); alt != obj {
		tc.Errorf(name.Pos(), "%s already declared at %s", name.Name, objPos(alt))
	}
	return obj
}


// decl is the same as declInScope(tc.topScope, ...)
func (tc *typechecker) decl(kind ast.Kind, name *ast.Ident, decl interface{}, n int) *ast.Object {
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
func (tc *typechecker) findField(typ *ast.Type, name *ast.Ident) (obj *ast.Object) {
	// TODO(gri) This is simplistic at the moment and ignores anonymous fields.
	obj = typ.Scope.Lookup(name.Name)
	if obj == nil {
		tc.Errorf(name.Pos(), "%s not declared", name.Name)
		obj = ast.NewObj(ast.Bad, name.Name)
	}
	return
}


// printScope prints the objects in a scope.
func printScope(scope *ast.Scope) {
	fmt.Printf("scope %p {", scope)
	if scope != nil && len(scope.Objects) > 0 {
		fmt.Println()
		for _, obj := range scope.Objects {
			form := "void"
			if obj.Type != nil {
				form = obj.Type.Form.String()
			}
			fmt.Printf("\t%s\t%s\n", obj.Name, form)
		}
	}
	fmt.Printf("}\n")
}
