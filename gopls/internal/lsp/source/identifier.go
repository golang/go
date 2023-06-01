// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"errors"
	"go/ast"
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// ErrNoIdentFound is error returned when no identifier is found at a particular position
var ErrNoIdentFound = errors.New("no identifier found")

// inferredSignature determines the resolved non-generic signature for an
// identifier in an instantiation expression.
//
// If no such signature exists, it returns nil.
func inferredSignature(info *types.Info, id *ast.Ident) *types.Signature {
	inst := typeparams.GetInstances(info)[id]
	sig, _ := inst.Type.(*types.Signature)
	return sig
}

func searchForEnclosing(info *types.Info, path []ast.Node) *types.TypeName {
	for _, n := range path {
		switch n := n.(type) {
		case *ast.SelectorExpr:
			if sel, ok := info.Selections[n]; ok {
				recv := Deref(sel.Recv())

				// Keep track of the last exported type seen.
				var exported *types.TypeName
				if named, ok := recv.(*types.Named); ok && named.Obj().Exported() {
					exported = named.Obj()
				}
				// We don't want the last element, as that's the field or
				// method itself.
				for _, index := range sel.Index()[:len(sel.Index())-1] {
					if r, ok := recv.Underlying().(*types.Struct); ok {
						recv = Deref(r.Field(index).Type())
						if named, ok := recv.(*types.Named); ok && named.Obj().Exported() {
							exported = named.Obj()
						}
					}
				}
				return exported
			}
		}
	}
	return nil
}

// typeToObject returns the relevant type name for the given type, after
// unwrapping pointers, arrays, slices, channels, and function signatures with
// a single non-error result, and ignoring built-in named types.
func typeToObject(typ types.Type) *types.TypeName {
	switch typ := typ.(type) {
	case *types.Named:
		// TODO(rfindley): this should use typeparams.NamedTypeOrigin.
		return typ.Obj()
	case *types.Pointer:
		return typeToObject(typ.Elem())
	case *types.Array:
		return typeToObject(typ.Elem())
	case *types.Slice:
		return typeToObject(typ.Elem())
	case *types.Chan:
		return typeToObject(typ.Elem())
	case *types.Signature:
		// Try to find a return value of a named type. If there's only one
		// such value, jump to its type definition.
		var res *types.TypeName

		results := typ.Results()
		for i := 0; i < results.Len(); i++ {
			obj := typeToObject(results.At(i).Type())
			if obj == nil || hasErrorType(obj) {
				// Skip builtins. TODO(rfindley): should comparable be handled here as well?
				continue
			}
			if res != nil {
				// The function/method must have only one return value of a named type.
				return nil
			}

			res = obj
		}
		return res
	default:
		return nil
	}
}

func hasErrorType(obj types.Object) bool {
	return types.IsInterface(obj.Type()) && obj.Pkg() == nil && obj.Name() == "error"
}

// typeSwitchImplicits returns all the implicit type switch objects that
// correspond to the leaf *ast.Ident. It also returns the original type
// associated with the identifier (outside of a case clause).
func typeSwitchImplicits(info *types.Info, path []ast.Node) ([]types.Object, types.Type) {
	ident, _ := path[0].(*ast.Ident)
	if ident == nil {
		return nil, nil
	}

	var (
		ts     *ast.TypeSwitchStmt
		assign *ast.AssignStmt
		cc     *ast.CaseClause
		obj    = info.ObjectOf(ident)
	)

	// Walk our ancestors to determine if our leaf ident refers to a
	// type switch variable, e.g. the "a" from "switch a := b.(type)".
Outer:
	for i := 1; i < len(path); i++ {
		switch n := path[i].(type) {
		case *ast.AssignStmt:
			// Check if ident is the "a" in "a := foo.(type)". The "a" in
			// this case has no types.Object, so check for ident equality.
			if len(n.Lhs) == 1 && n.Lhs[0] == ident {
				assign = n
			}
		case *ast.CaseClause:
			// Check if ident is a use of "a" within a case clause. Each
			// case clause implicitly maps "a" to a different types.Object,
			// so check if ident's object is the case clause's implicit
			// object.
			if obj != nil && info.Implicits[n] == obj {
				cc = n
			}
		case *ast.TypeSwitchStmt:
			// Look for the type switch that owns our previously found
			// *ast.AssignStmt or *ast.CaseClause.
			if n.Assign == assign {
				ts = n
				break Outer
			}

			for _, stmt := range n.Body.List {
				if stmt == cc {
					ts = n
					break Outer
				}
			}
		}
	}
	if ts == nil {
		return nil, nil
	}
	// Our leaf ident refers to a type switch variable. Fan out to the
	// type switch's implicit case clause objects.
	var objs []types.Object
	for _, cc := range ts.Body.List {
		if ccObj := info.Implicits[cc]; ccObj != nil {
			objs = append(objs, ccObj)
		}
	}
	// The right-hand side of a type switch should only have one
	// element, and we need to track its type in order to generate
	// hover information for implicit type switch variables.
	var typ types.Type
	if assign, ok := ts.Assign.(*ast.AssignStmt); ok && len(assign.Rhs) == 1 {
		if rhs := assign.Rhs[0].(*ast.TypeAssertExpr); ok {
			typ = info.TypeOf(rhs.X)
		}
	}
	return objs, typ
}
