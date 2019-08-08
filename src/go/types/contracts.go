// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type-checking of contracts.

package types

import (
	"go/ast"
	"sort"
)

// TODO(gri) Handling a contract like a type is problematic because it
// won't exclude a contract where we only permit a type. Investigate.

func (check *Checker) contractType(contr *Contract, e *ast.ContractType) {
	check.openScope(e, "contract")
	defer check.closeScope()

	// collect type parameters
	tparams := make([]*TypeName, len(e.TParams))
	for index, name := range e.TParams {
		tpar := NewTypeName(name.Pos(), check.pkg, name.Name, nil)
		NewTypeParam(tpar, index, nil) // assigns type to tpar as a side-effect
		check.declare(check.scope, name, tpar, check.scope.pos)
		tparams[index] = tpar
	}

	// each type parameter's constraints are represented by an interface
	ifaces := make(map[*TypeName]*Interface)

	addMethod := func(tpar *TypeName, m *Func) {
		iface := ifaces[tpar]
		if iface == nil {
			iface = new(Interface)
			ifaces[tpar] = iface
		}
		iface.methods = append(iface.methods, m)
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
			switch nmethods {
			case 0:
				// type constraints
				for _, texpr := range c.Types {
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
					// TODO(gri) add type
				}

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
				addMethod(tpar, m)

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

	// cleanup/complete interfaces
	// TODO(gri) should check for duplicate entries in first pass (=> no need for this extra pass)
	for tpar, iface := range ifaces {
		if iface == nil {
			ifaces[tpar] = &emptyInterface
		} else {
			var mset objset
			i := 0
			for _, m := range iface.methods {
				if m0 := mset.insert(m); m0 != nil {
					// A method with the same name exists already.
					check.errorf(m.Pos(), "method %s already declared", m.name)
					check.reportAltDecl(m0)
				} else {
					// only keep unique methods
					// TODO(gri) revisit this code - introduced to fix large rebase
					iface.methods[i] = m
					i++
				}
			}
			iface.methods = iface.methods[:i]
			sort.Sort(byUniqueMethodName(iface.methods))
			iface.Complete()
		}
	}

	contr.TParams = tparams
	contr.IFaces = ifaces
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
		// ok
	default:
		unreachable()
	}
	return true
}

// satisfyContract reports whether the given type arguments satisfy a contract.
// The contract may be nil, in which case it is always satisfied.
// The number of type arguments must match the number of contract type parameters.
// TODO(gri) missing: good error reporting
func (check *Checker) satisfyContract(contr *Contract, targs []Type) bool {
	if contr == nil {
		return true
	}

	assert(len(contr.TParams) == len(targs))

	// A contract is represented by a list of interfaces, one for each
	// contract type parameter. Each of those interfaces may be parameterized
	// with any of the other contract type parameters.
	// We need to verify that each type argument implements its respective
	// contract interface, but only after substituting any contract type parameters
	// in that interface with the respective type arguments.
	//
	// TODO(gri) This implementation strategy (and implementation) would be
	// much more direct if we were to replace contracts simply with (parameterized)
	// interfaces. All the existing machinery would simply fall into place.

	for i, targ := range targs {
		iface := contr.ifaceAt(i)
		if iface == nil {
			continue // no constraints
		}
		// If iface is parameterized, we need to replace the type parameters
		// with the respective type arguments.
		if isParameterized(iface) {
			panic("unimplemented")
		}
		// targ must implement iface
		if m, _ := check.missingMethod(targ, iface, true); m != nil {
			return false
		}
	}

	return true
}
