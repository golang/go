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

func (check *Checker) contractType(contr *Contract, name string, e *ast.ContractType) {
	check.openScope(e, "contract")
	defer check.closeScope()

	// collect type parameters
	tparams := make([]*TypeName, len(e.TParams))
	for index, name := range e.TParams {
		tpar := NewTypeName(name.Pos(), check.pkg, name.Name, nil)
		check.NewTypeParam(tpar, index, nil) // assigns type to tpar as a side-effect
		check.declare(check.scope, name, tpar, check.scope.pos)
		tparams[index] = tpar
	}

	// TODO(gri) review this - we probably don't need lazy allocation anymore
	// Each type parameter's constraints are represented by a (lazily allocated) named interface.
	// Given a contract C(P1, P2, ... Pn) { ... } we construct named types C1(P1, P2, ... Pn),
	// C2(P1, P2, ... Pn), ... Cn(P1, P2, ... Pn) with the respective underlying interfaces
	// representing the type constraints for each of the type parameters (C1 for P1, C2 for P2, etc.).
	bounds := make(map[*TypeName]*Named)
	ifaceFor := func(tpar *TypeName) *Interface {
		named := bounds[tpar]
		if named == nil {
			index := tpar.typ.(*TypeParam).index
			tname := NewTypeName(e.Pos(), check.pkg, name+string(subscript(uint64(index))), nil)
			tname.tparams = tparams
			named = NewNamed(tname, new(Interface), nil)
			bounds[tpar] = named
		}
		return named.underlying.(*Interface)
	}

	// collect constraints
	for _, c := range e.Constraints {
		if c.Param != nil {
			// If a type name is present, it must be one of the contract's type parameters.
			pos := c.Param.Pos()
			obj := check.scope.Lookup(c.Param.Name)
			if obj == nil {
				check.errorf(pos, "%s not declared by contract", c.Param.Name)
				continue
			}
			if c.Types == nil {
				check.invalidAST(pos, "missing method or type constraint")
				continue
			}

			// For now we only allow a single method or a list of types,
			// and not multiple methods or a mix of methods and types.
			nmethods := 0 // must be 0 or 1 or we have an error
			for i, mname := range c.MNames {
				if mname != nil {
					nmethods++
					if nmethods > 1 {
						check.errorf(mname.Pos(), "cannot have more than one method")
						break
					}
				} else if nmethods > 0 {
					nmethods = 2 // mark as invalid
					pos := pos   // fallback position in case we don't have a type
					if i < len(c.Types) && c.Types[i] != nil {
						pos = c.Types[i].Pos()
					}
					check.errorf(pos, "cannot mix types and methods")
					break
				}
			}

			tpar := obj.(*TypeName)
			iface := ifaceFor(tpar)
			switch nmethods {
			case 0:
				// type constraints
				iface.types = check.collectTypeConstraints(pos, iface.types, c.Types)

			case 1:
				// method constraint
				if nmethods != len(c.Types) {
					check.invalidAST(pos, "number of method names and signatures doesn't match")
				}
				// If a method is present, it must be unique for the respective type
				// parameter, and c.Types[0] is a method signature (guaranteed by AST).
				typ := check.typ(c.Types[0])
				sig, _ := typ.(*Signature)
				if sig == nil {
					check.invalidAST(c.Types[0].Pos(), "invalid method type %s", typ)
				}
				// add receiver to signature
				// (TODO(gri) verify that this matches what we do elsewhere, e.g., in NewInterfaceType)
				assert(sig.recv == nil)
				recvTyp := tpar.typ
				sig.recv = NewVar(pos, check.pkg, "", recvTyp)
				// add the method
				mname := c.MNames[0]
				m := NewFunc(mname.Pos(), check.pkg, mname.Name, sig)
				iface.methods = append(iface.methods, m)

			default:
				// ignore (error was reported earlier)
			}

		} else {

			// no type name => we have an embedded contract
			// A correct AST will have no method name and a single type that is an *ast.CallExpr in this case.
			if len(c.MNames) != 0 {
				check.invalidAST(c.MNames[0].Pos(), "no method (%s) expected with embedded contract declaration", c.MNames[0].Name)
				// ignore and continue
			}
			if len(c.Types) != 1 {
				check.invalidAST(e.Pos(), "contract contains incorrect (possibly embedded contract) entry")
				continue
			}
			// TODO(gri) we can probably get away w/o checking this (even if the AST is broken)
			econtr, _ := c.Types[0].(*ast.CallExpr)
			if econtr == nil {
				check.invalidAST(c.Types[0].Pos(), "invalid embedded contract %s", econtr)
			}
			etyp := check.typ(c.Types[0])
			_ = etyp
			// TODO(gri) complete this
			check.errorf(c.Types[0].Pos(), "%s: contract embedding not yet implemented", c.Types[0])
		}
	}

	// complete interfaces
	for _, bound := range bounds {
		check.completeInterface(e.Pos(), bound.underlying.(*Interface))
	}

	contr.TParams = tparams
	contr.Bounds = bounds
}

func (check *Checker) collectTypeConstraints(pos token.Pos, list []Type, types []ast.Expr) []Type {
	for _, texpr := range types {
		if texpr == nil {
			check.invalidAST(pos, "missing type constraint")
			continue
		}
		typ := check.typ(texpr)
		// A type constraint may be a predeclared type or a
		// composite type composed only of predeclared types.
		// TODO(gri) should we keep this restriction?
		var why string
		if !check.typeConstraint(typ, &why) {
			check.errorf(texpr.Pos(), "invalid type constraint %s (%s)", typ, why)
			continue
		}
		// add type
		list = append(list, typ)
	}
	return list
}

// TODO(gri) does this simply check for the absence of defined types?
//           (if so, should choose a better name)
func (check *Checker) typeConstraint(typ Type, why *string) bool {
	switch t := typ.(type) {
	case *Basic:
		// ok
	case *Array:
		return check.typeConstraint(t.elem, why)
	case *Slice:
		return check.typeConstraint(t.elem, why)
	case *Struct:
		for _, f := range t.fields {
			if !check.typeConstraint(f.typ, why) {
				return false
			}
		}
	case *Pointer:
		return check.typeConstraint(t.base, why)
	case *Tuple:
		if t == nil {
			return true
		}
		for _, v := range t.vars {
			if !check.typeConstraint(v.typ, why) {
				return false
			}
		}
	case *Signature:
		if len(t.tparams) != 0 {
			panic("type parameter in function type")
		}
		return (t.recv == nil || check.typeConstraint(t.recv.typ, why)) &&
			check.typeConstraint(t.params, why) &&
			check.typeConstraint(t.results, why)
	case *Interface:
		for _, m := range t.allMethods {
			if !check.typeConstraint(m.typ, why) {
				return false
			}
		}
	case *Map:
		return check.typeConstraint(t.key, why) && check.typeConstraint(t.elem, why)
	case *Chan:
		return check.typeConstraint(t.elem, why)
	case *Named:
		*why = check.sprintf("%s is not a type literal", t)
		return false
	case *Contract:
		// TODO(gri) we shouldn't reach here
		*why = check.sprintf("%s is not a type", t)
		return false
	case *TypeParam:
		// TODO(gri) should this be ok? need a good use case
		// ok for now
	default:
		unreachable()
	}
	return true
}
