// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type parameter inference given
// a list of concrete arguments and a parameter list.

package types

import (
	"go/token"
	"strings"
)

// infer returns the list of actual type arguments for the given list of type parameters tparams
// by inferring them from the actual arguments args for the parameters params. If type inference
// is impossible because unification fails, an error is reported and the resulting types list is
// nil, and index is 0. Otherwise, types is the list of inferred type arguments, and index is
// the index of the first type argument in that list that couldn't be inferred (and thus is nil).
// If all type arguments where inferred successfully, index is < 0.
func (check *Checker) infer(tparams []*TypeName, params *Tuple, args []*operand) (types []Type, index int) {
	assert(params.Len() == len(args))

	u := newUnifier(check, false)
	u.x.init(tparams)

	errorf := func(kind string, tpar, targ Type, arg *operand) {
		// provide a better error message if we can
		targs, failed := u.x.types()
		if failed == 0 {
			// The first type parameter couldn't be inferred.
			// If none of them could be inferred, don't try
			// to provide the inferred type in the error msg.
			allFailed := true
			for _, targ := range targs {
				if targ != nil {
					allFailed = false
					break
				}
			}
			if allFailed {
				check.errorf(arg.pos(), "%s %s of %s does not match %s (cannot infer %s)", kind, targ, arg.expr, tpar, typeNamesString(tparams))
				return
			}
		}
		smap := makeSubstMap(tparams, targs)
		inferred := check.subst(arg.pos(), tpar, smap)
		if inferred != tpar {
			check.errorf(arg.pos(), "%s %s of %s does not match inferred type %s for %s", kind, targ, arg.expr, inferred, tpar)
		} else {
			check.errorf(arg.pos(), "%s %s of %s does not match %s", kind, targ, arg.expr, tpar)
		}
	}

	// Terminology: generic parameter = function parameter with a type-parameterized type

	// 1st pass: Unify parameter and argument types for generic parameters with typed arguments
	//           and collect the indices of generic parameters with untyped arguments.
	var indices []int
	for i, arg := range args {
		par := params.At(i)
		// If we permit bidirectional unification, this conditional code needs to be
		// executed even if par.typ is not parameterized since the argument may be a
		// generic function (for which we want to infer // its type arguments).
		if IsParameterized(par.typ) {
			if arg.mode == invalid {
				// An error was reported earlier. Ignore this targ
				// and continue, we may still be able to infer all
				// targs resulting in fewer follon-on errors.
				continue
			}
			if targ := arg.typ; isTyped(targ) {
				// If we permit bidirectional unification, and targ is
				// a generic function, we need to initialize u.y with
				// the respectice type parameters of targ.
				if !u.unify(par.typ, targ) {
					errorf("type", par.typ, targ, arg)
					return nil, 0
				}
			} else {
				indices = append(indices, i)
			}
		}
	}

	// Some generic parameters with untyped arguments may have been given a type
	// indirectly through another generic parameter with a typed argument; we can
	// ignore those now. (This only means that we know the types for those generic
	// parameters; it doesn't mean untyped arguments can be passed safely. We still
	// need to verify that assignment of those arguments is valid when we check
	// function parameter passing external to infer.)
	j := 0
	for _, i := range indices {
		par := params.At(i)
		// Since untyped types are all basic (i.e., non-composite) types, an
		// untyped argument will never match a composite parameter type; the
		// only parameter type it can possibly match against is a *TypeParam.
		// Thus, only keep the indices of generic parameters that are not of
		// composite types and which don't have a type inferred yet.
		if tpar, _ := par.typ.(*TypeParam); tpar != nil && u.x.at(tpar.index) == nil {
			indices[j] = i
			j++
		}
	}
	indices = indices[:j]

	// 2nd pass: Unify parameter and default argument types for remaining generic parameters.
	for _, i := range indices {
		par := params.At(i)
		arg := args[i]
		targ := Default(arg.typ)
		// The default type for an untyped nil is untyped nil. We must not
		// infer an untyped nil type as type parameter type. Ignore untyped
		// nil by making sure all default argument types are typed.
		if isTyped(targ) && !u.unify(par.typ, targ) {
			errorf("default type", par.typ, targ, arg)
			return nil, 0
		}
	}

	return u.x.types()
}

// typeNamesString produces a string containing all the
// type names in list suitable for human consumption.
func typeNamesString(list []*TypeName) string {
	// common cases
	n := len(list)
	switch n {
	case 0:
		return ""
	case 1:
		return list[0].name
	case 2:
		return list[0].name + " and " + list[1].name
	}

	// general case (n > 2)
	var b strings.Builder
	for i, tname := range list[:n-1] {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(tname.name)
	}
	b.WriteString(", and ")
	b.WriteString(list[n-1].name)
	return b.String()
}

// IsParameterized reports whether typ contains any type parameters.
// TODO(gri) This is not strictly correct. We only want the free
// type parameters for a given type. (At the moment, the only way
// to mix free and bound type parameters is through method type parameters
// on parameterized receiver types - need to investigate.)
func IsParameterized(typ Type) bool {
	return isParameterized(typ, make(map[Type]bool))
}

func isParameterized(typ Type, seen map[Type]bool) (res bool) {
	// detect cycles
	// TODO(gri) can/should this be a Checker map?
	if x, ok := seen[typ]; ok {
		return x
	}
	seen[typ] = false
	defer func() {
		seen[typ] = res
	}()

	switch t := typ.(type) {
	case nil, *Basic: // TODO(gri) should nil be handled here?
		break

	case *Array:
		return isParameterized(t.elem, seen)

	case *Slice:
		return isParameterized(t.elem, seen)

	case *Struct:
		for _, fld := range t.fields {
			if isParameterized(fld.typ, seen) {
				return true
			}
		}

	case *Pointer:
		return isParameterized(t.base, seen)

	case *Tuple:
		n := t.Len()
		for i := 0; i < n; i++ {
			if isParameterized(t.At(i).typ, seen) {
				return true
			}
		}

	case *Sum:
		return isParameterizedList(t.types, seen)

	case *Signature:
		assert(t.tparams == nil) // TODO(gri) is this correct?
		// TODO(gri) Rethink check below.
		//assert(t.recv == nil || !isParameterized(t.recv.typ))
		return isParameterized(t.params, seen) || isParameterized(t.results, seen)

	case *Interface:
		if t.allMethods != nil {
			// interface is complete - quick test
			for _, m := range t.allMethods {
				if isParameterized(m.typ, seen) {
					return true
				}
			}
			return isParameterizedList(unpack(t.allTypes), seen)
		}

		return t.iterate(func(t *Interface) bool {
			for _, m := range t.methods {
				if isParameterized(m.typ, seen) {
					return true
				}
			}
			return isParameterizedList(unpack(t.types), seen)
		}, nil)

	case *Map:
		return isParameterized(t.key, seen) || isParameterized(t.elem, seen)

	case *Chan:
		return isParameterized(t.elem, seen)

	case *Named:
		return isParameterizedList(t.targs, seen)

	case *TypeParam:
		return true

	case *instance:
		return isParameterizedList(t.targs, seen)

	default:
		unreachable()
	}

	return false
}

// IsParameterizedList reports whether any type in list is parameterized.
func IsParameterizedList(list []Type) bool {
	return isParameterizedList(list, make(map[Type]bool))
}

func isParameterizedList(list []Type, seen map[Type]bool) bool {
	for _, t := range list {
		if isParameterized(t, seen) {
			return true
		}
	}
	return false
}

// inferB returns the list of actual type arguments inferred from the type parameters'
// bounds and an initial set of type arguments. If type inference is impossible because
// unification fails, an error is reported, the resulting types list is nil, and index is 0.
// Otherwise, types is the list of inferred type arguments, and index is the index of the
// first type argument in that list that couldn't be inferred (and thus is nil). If all
// type arguments where inferred successfully, index is < 0. The number of type arguments
// provided may be less than the number of type parameters, but there must be at least one.
func (check *Checker) inferB(tparams []*TypeName, targs []Type) (types []Type, index int) {
	assert(len(tparams) >= len(targs) && len(targs) > 0)

	// Setup bidirectional unification between those structural bounds
	// and the corresponding type arguments (which may be nil!).
	u := newUnifier(check, false)
	u.x.init(tparams)
	u.y = u.x // type parameters between LHS and RHS of unification are identical

	// Set the type arguments which we know already.
	for i, targ := range targs {
		if targ != nil {
			u.x.set(i, targ)
		}
	}

	// Unify type parameters with their structural constraints, if any.
	for _, tpar := range tparams {
		typ := tpar.typ.(*TypeParam)
		sbound := check.structuralType(typ.bound.Under())
		if sbound != nil {
			//check.dump(">>> unify(%s, %s)", tpar, sbound)
			if !u.unify(typ, sbound) {
				check.errorf(tpar.pos, "%s does not match %s", tpar, sbound)
				return nil, 0
			}
			//check.dump(">>> => indices = %v, types = %s", u.x.indices, u.types)
		}
	}

	// u.x.types() now contains the incoming type arguments plus any additional type
	// arguments for which there were structural constraints. The newly inferred non-
	// nil entries may still contain references to other type parameters. For instance,
	// for [type A interface{}, B interface{type []C}, C interface{type *A}], if A == int
	// was given, unification produced the type list [int, []C, *A]. We eliminate the
	// remaining type parameters by substituting the type parameters in this type list
	// until nothing changes anymore.
	types, index = u.x.types()
	if debug {
		for i, targ := range targs {
			assert(targ == nil || types[i] == targ)
		}
	}

	// dirty tracks the indices of all types that may still contain type parameters.
	// We know that nil types entries and entries corresponding to provided (non-nil)
	// type arguments are clean, so exclude them from the start.
	var dirty []int
	for i, typ := range types {
		if typ != nil && (i >= len(targs) || targs[i] == nil) {
			dirty = append(dirty, i)
		}
	}

	for len(dirty) > 0 {
		// TODO(gri) Instead of creating a new smap for each iteration,
		// provide an update operation for smaps and only change when
		// needed. Optimization.
		smap := makeSubstMap(tparams, types)
		n := 0
		for _, index := range dirty {
			t0 := types[index]
			if t1 := check.subst(token.NoPos, t0, smap); t1 != t0 {
				types[index] = t1
				dirty[n] = index
				n++
			}
		}
		dirty = dirty[:n]
	}
	//check.dump(">>> inferred types = %s", types)

	return
}

// structuralType returns the structural type of a constraint, if any.
func (check *Checker) structuralType(constraint Type) Type {
	if iface, _ := constraint.(*Interface); iface != nil {
		check.completeInterface(token.NoPos, iface)
		types := unpack(iface.allTypes)
		if len(types) == 1 {
			return types[0]
		}
		return nil
	}
	return constraint
}
