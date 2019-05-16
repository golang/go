// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type-checking of contracts.

package types

import (
	"go/ast"
	"go/token"
)

// TODO(gri) Handling a contract like a type is problematic because it
// won't exclude a contract where we only permit a type. Investigate.

func (check *Checker) contractType(ctyp *Contract, e *ast.ContractType) {
	scope := NewScope(check.scope, token.NoPos, token.NoPos, "contract type parameters")
	check.scope = scope
	defer check.closeScope()
	check.recordScope(e, scope)

	// collect type parameters
	var tparams []*TypeName
	for index, name := range e.TParams {
		tpar := NewTypeName(name.Pos(), check.pkg, name.Name, nil)
		NewTypeParam(tpar, index) // assigns type to tpar as a side-effect
		check.declare(scope, name, tpar, scope.pos)
		tparams = append(tparams, tpar)
	}
	ctyp.TParams = tparams

	// collect constraints
	for _, c := range e.Constraints {
		if c.Param != nil {
			// If a type name is present, it must be one of the contract's type parameters.
			tpar := scope.Lookup(c.Param.Name)
			if tpar == nil {
				check.errorf(c.Param.Pos(), "%s not declared by contract", c.Param.Name)
				continue // TODO(gri) should try fall through
			}
			if c.Type == nil {
				check.invalidAST(c.Param.Pos(), "missing method or type constraint")
				continue
			}
			typ := check.typ(c.Type)
			if c.MName != nil {
				// If a method name is present, it must be unique for the respective
				// type parameter, and c.Type is a method signature (guaranteed by AST).
				sig, _ := typ.(*Signature)
				if sig == nil {
					check.invalidAST(c.Type.Pos(), "invalid method type")
				}
				// TODO(gri) what requirements do we have for sig.scope, sig.recv?
			} else {
				// no method name => we have a type constraint
				var why string
				if !check.constraint(typ, &why) {
					check.errorf(c.Type.Pos(), "invalid type constraint %s (%s)", typ, why)
					continue
				}
			}
		} else {
			// no type name => we have an embedded contract
			panic("embedded contracts unimplemented")
		}
	}
}

func (check *Checker) constraint(typ Type, why *string) bool {
	switch t := typ.(type) {
	case *Basic:
		// ok
	case *Array:
		return check.constraint(t.elem, why)
	case *Slice:
		return check.constraint(t.elem, why)
	case *Struct:
		for _, f := range t.fields {
			if !check.constraint(f.typ, why) {
				return false
			}
		}
	case *Pointer:
		return check.constraint(t.base, why)
	case *Tuple:
		panic("tuple type checking unimplemented")
	case *Signature:
		panic("signature type checking unimplemented")
	case *Interface:
		panic("interface type checking unimplemented")
	case *Map:
		return check.constraint(t.key, why) && check.constraint(t.elem, why)
	case *Chan:
		return check.constraint(t.elem, why)
	case *Named:
		*why = check.sprintf("%s is not a type literal", t)
		return false
	case *Contract:
		*why = check.sprintf("%s is not a type", t)
		return false
	case *TypeParam:
		// ok
	default:
		unreachable()
	}
	return true
}
