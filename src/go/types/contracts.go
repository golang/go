// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type-checking of contracts.

package types

import (
	"go/ast"
	"go/token"
)

func (check *Checker) contractDecl(obj *Contract, cdecl *ast.ContractSpec) {
	assert(obj.typ == nil)

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
				check.errorf(pos, "%s is not a type parameter declared by the contract", c.Param.Name)
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

			ifaceName := bounds[tobj.(*TypeName).typ.(*TypeParam).index]
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
				continue
			}

			// Handle contract lookup here so we don't need to set up a special contract mode
			// for operands just to carry its information through in form of some contract Type.
			// TODO(gri) this code is also in collectTypeParams (decl.go) - factor out!
			if ident, ok := unparen(econtr.Fun).(*ast.Ident); ok {
				if eobj, _ := check.lookup(ident.Name).(*Contract); eobj != nil {
					// set up contract if not yet done
					if eobj.typ == nil {
						check.objDecl(eobj, nil)
						if eobj.typ == Typ[Invalid] {
							continue // don't use contract
						}
					}
					// eobj is a valid contract
					// contract arguments must match the embedded contract's parameters
					n := len(econtr.Args)
					if n != len(eobj.TParams) {
						check.errorf(c.Types[0].Pos(), "%d type parameters but contract expects %d", n, len(eobj.TParams))
						continue
					}

					// contract arguments must be type parameters from the enclosing contract
					var targs []Type
					for _, arg := range econtr.Args {
						targ := check.typ(arg)
						if parg, _ := targ.(*TypeParam); parg != nil {
							// Contract declarations are only permitted at the package level,
							// thus the only type parameters visible inside a contract are
							// type parameters from the enclosing contract.
							// The assertion below cannot fail unless we change the premise
							// and permit contract declarations inside parameterized functions.
							assert(tparams[parg.index] == parg.obj)
							targs = append(targs, targ)
						} else if targ != Typ[Invalid] {
							check.errorf(arg.Pos(), "%s is not a type parameter", arg)
						}
					}
					if len(targs) < n {
						continue // some arguments were incorrect
					}

					// instantiate each (embedded) contract bound with contract arguments
					assert(n == len(eobj.Bounds))
					ebounds := make([]*Named, n)
					for i, ebound := range eobj.Bounds {
						ebounds[i] = check.instantiate(econtr.Args[i].Pos(), ebound, targs, nil).(*Named)
					}

					// add the instantiated bounds as embedded interfaces to the respective
					// embedding (outer) contract bound
					for i, ebound := range ebounds {
						index := targs[i].(*TypeParam).index
						iface := bounds[index].underlying.(*Interface)
						iface.embeddeds = append(iface.embeddeds, ebound)
						check.posMap[iface] = append(check.posMap[iface], econtr.Pos()) // satisfy completeInterface requirements
					}
				}
				continue // success
			}

			check.errorf(c.Types[0].Pos(), "%s is not a contract", c.Types[0])
		}
	}

	// complete interfaces
	for _, bound := range bounds {
		check.completeInterface(cdecl.Pos(), bound.underlying.(*Interface))
	}

	obj.typ = new(contractType) // mark contract as fully set up
	obj.TParams = tparams
	obj.Bounds = bounds
}

// Contracts don't have types, but we need to set a type to
// detect recursive declarations and satisfy assertions.
type contractType struct{}

func (contractType) String() string   { return "<dummy contract type>" }
func (contractType) Underlying() Type { panic("unreachable") }

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
		t.assertCompleteness()
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
