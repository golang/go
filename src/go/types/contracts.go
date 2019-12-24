// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type-checking of contracts.

package types

import (
	"go/ast"
	"go/token"
)

type contractType struct{}

func (contractType) String() string   { return "<dummy contract type>" }
func (contractType) Underlying() Type { panic("unreachable") }

func (check *Checker) contractDecl(obj *Contract, cdecl *ast.ContractSpec) {
	assert(obj.typ == nil)

	// contracts don't have types, but we need to set a type to
	// detect recursive declarations and satisfy various assertions
	obj.typ = new(contractType)

	check.openScope(cdecl, "contract")
	defer check.closeScope()

	tparams := check.declareTypeParams(nil, cdecl.TParams, nil)

	// Given a contract C(P1, P2, ... Pn) { ... } we construct named types C1(P1, P2, ... Pn),
	// C2(P1, P2, ... Pn), ... Cn(P1, P2, ... Pn) with the respective underlying interfaces
	// representing the (possibly empty) type constraints for each of the type parameters
	// (C1 for P1, C2 for P2, etc.).
	bounds := make([]*Named, len(tparams))
	for i, tpar := range tparams {
		tname := NewTypeName(tpar.Pos(), check.pkg, obj.name+string(subscript(uint64(i))), nil)
		named := NewNamed(tname, new(Interface), nil)
		named.tparams = tparams
		bounds[i] = named
	}

	// collect constraints
	for _, c := range cdecl.Constraints {
		if c.Param != nil {
			// If a type name is present, it must be one of the contract's type parameters.
			pos := c.Param.Pos()
			tobj := check.scope.Lookup(c.Param.Name)
			if tobj == nil {
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

			tpar := tobj.(*TypeName)
			ifaceName := bounds[tpar.typ.(*TypeParam).index]
			iface := ifaceName.underlying.(*Interface)
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
				sig.recv = NewVar(pos, check.pkg, "_", ifaceName)
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
				check.invalidAST(cdecl.Pos(), "contract contains incorrect (possibly embedded contract) entry")
				continue
			}
			// TODO(gri) we can probably get away w/o checking this (even if the AST is broken)
			econtr, _ := c.Types[0].(*ast.CallExpr)
			if econtr == nil {
				check.invalidAST(c.Types[0].Pos(), "invalid embedded contract %s", econtr)
			}
			// Handle contract lookup so we don't need to set up a special contract mode
			// for operands just to carry its information through in form of some contract Type.
			// TODO(gri) this code is also in collectTypeParams (decl.go) - factor out!
			if ident, ok := unparen(econtr.Fun).(*ast.Ident); ok {
				if eobj, _ := check.lookup(ident.Name).(*Contract); eobj != nil {
					// TODO(gri) must set up contract if not yet done!
					// eobj is a valid contract
					// TODO(gri) look for contract cycles!
					// contract arguments must match the embedded contract's parameters
					if len(econtr.Args) != len(eobj.TParams) {
						check.errorf(c.Types[0].Pos(), "%d type parameters but contract expects %d", len(econtr.Args), len(eobj.TParams))
						continue
					}
					// contract arguments must be type parameters
					// For now, they must be from the enclosing contract.
					// TODO(gri) can we allow any type parameter?
					targs := make([]Type, len(econtr.Args))
					for i, arg := range econtr.Args {
						targ := check.typ(arg)
						if parg, _ := targ.(*TypeParam); parg != nil {
							// TODO(gri) check that targ is a parameter from the enclosing contract
							targs[i] = targ
						} else {
							check.errorf(arg.Pos(), "%s is not a type parameter", arg)
						}
					}
					// TODO(gri) - implement the steps below
					// - for each eobj type parameter, determine its (interface) bound
					// - substitute that type parameter with the actual type argument in that interface
					// - add the interface as am embedded interface to the bound matching the actual type argument
					// - tests! (incl. overlapping methods, etc.)
					check.errorf(c.Types[0].Pos(), "%s: contract embedding not yet implemented", c.Types[0])
					continue
				}
			}
			check.errorf(c.Types[0].Pos(), "%s is not a contract", c.Types[0])
		}
	}

	// complete interfaces
	for _, bound := range bounds {
		check.completeInterface(cdecl.Pos(), bound.underlying.(*Interface))
	}

	obj.TParams = tparams
	obj.Bounds = bounds
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
	case *TypeParam:
		// ok, e.g.: func f (type T interface { type T }) ()
	default:
		unreachable()
	}
	return true
}
