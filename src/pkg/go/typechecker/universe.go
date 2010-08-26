// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typechecker

import "go/ast"

// TODO(gri) should this be in package ast?

// The Universe scope contains all predeclared identifiers.
var Universe *ast.Scope


func def(obj *ast.Object) {
	alt := Universe.Insert(obj)
	if alt != obj {
		panic("object declared twice")
	}
}


func init() {
	Universe = ast.NewScope(nil)

	// basic types
	for n, name := range ast.BasicTypes {
		typ := ast.NewType(ast.Basic)
		typ.N = n
		obj := ast.NewObj(ast.Typ, name)
		obj.Type = typ
		typ.Obj = obj
		def(obj)
	}

	// built-in functions
	// TODO(gri) implement this
}
