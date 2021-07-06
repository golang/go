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

// infer attempts to infer the complete set of type arguments for generic function instantiation/call
// based on the given type parameters tparams, type arguments targs, function parameters params, and
// function arguments args, if any. There must be at least one type parameter, no more type arguments
// than type parameters, and params and args must match in number (incl. zero).
// If successful, infer returns the complete list of type arguments, one for each type parameter.
// Otherwise the result is nil and appropriate errors will be reported unless report is set to false.
//
// Inference proceeds in 3 steps:
//
//   1) Start with given type arguments.
//   2) Infer type arguments from typed function arguments.
//   3) Infer type arguments from untyped function arguments.
//
// Constraint type inference is used after each step to expand the set of type arguments.
//
func (check *Checker) infer(posn positioner, tparams []*TypeName, targs []Type, params *Tuple, args []*operand, report bool) (result []Type) {
	if debug {
		defer func() {
			assert(result == nil || len(result) == len(tparams))
			for _, targ := range result {
				assert(targ != nil)
			}
			//check.dump("### inferred targs = %s", result)
		}()
	}

	// There must be at least one type parameter, and no more type arguments than type parameters.
	n := len(tparams)
	assert(n > 0 && len(targs) <= n)

	// Function parameters and arguments must match in number.
	assert(params.Len() == len(args))

	// --- 0 ---
	// If we already have all type arguments, we're done.
	if len(targs) == n {
		return targs
	}
	// len(targs) < n

	// --- 1 ---
	// Explicitly provided type arguments take precedence over any inferred types;
	// and types inferred via constraint type inference take precedence over types
	// inferred from function arguments.
	// If we have type arguments, see how far we get with constraint type inference.
	if len(targs) > 0 {
		var index int
		targs, index = check.inferB(tparams, targs, report)
		if targs == nil || index < 0 {
			return targs
		}
	}

	// Continue with the type arguments we have now. Avoid matching generic
	// parameters that already have type arguments against function arguments:
	// It may fail because matching uses type identity while parameter passing
	// uses assignment rules. Instantiate the parameter list with the type
	// arguments we have, and continue with that parameter list.

	// First, make sure we have a "full" list of type arguments, so of which
	// may be nil (unknown).
	if len(targs) < n {
		targs2 := make([]Type, n)
		copy(targs2, targs)
		targs = targs2
	}
	// len(targs) == n

	// Substitute type arguments for their respective type parameters in params,
	// if any. Note that nil targs entries are ignored by check.subst.
	// TODO(gri) Can we avoid this (we're setting known type argumemts below,
	//           but that doesn't impact the isParameterized check for now).
	if params.Len() > 0 {
		smap := makeSubstMap(tparams, targs)
		params = check.subst(token.NoPos, params, smap).(*Tuple)
	}

	// --- 2 ---
	// Unify parameter and argument types for generic parameters with typed arguments
	// and collect the indices of generic parameters with untyped arguments.
	// Terminology: generic parameter = function parameter with a type-parameterized type
	u := newUnifier(check, false)
	u.x.init(tparams)

	// Set the type arguments which we know already.
	for i, targ := range targs {
		if targ != nil {
			u.x.set(i, targ)
		}
	}

	errorf := func(kind string, tpar, targ Type, arg *operand) {
		if !report {
			return
		}
		// provide a better error message if we can
		targs, index := u.x.types()
		if index == 0 {
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
				check.errorf(arg, _Todo, "%s %s of %s does not match %s (cannot infer %s)", kind, targ, arg.expr, tpar, typeNamesString(tparams))
				return
			}
		}
		smap := makeSubstMap(tparams, targs)
		// TODO(rFindley): pass a positioner here, rather than arg.Pos().
		inferred := check.subst(arg.Pos(), tpar, smap)
		if inferred != tpar {
			check.errorf(arg, _Todo, "%s %s of %s does not match inferred type %s for %s", kind, targ, arg.expr, inferred, tpar)
		} else {
			check.errorf(arg, 0, "%s %s of %s does not match %s", kind, targ, arg.expr, tpar)
		}
	}

	// indices of the generic parameters with untyped arguments - save for later
	var indices []int
	for i, arg := range args {
		par := params.At(i)
		// If we permit bidirectional unification, this conditional code needs to be
		// executed even if par.typ is not parameterized since the argument may be a
		// generic function (for which we want to infer its type arguments).
		if isParameterized(tparams, par.typ) {
			if arg.mode == invalid {
				// An error was reported earlier. Ignore this targ
				// and continue, we may still be able to infer all
				// targs resulting in fewer follon-on errors.
				continue
			}
			if targ := arg.typ; isTyped(targ) {
				// If we permit bidirectional unification, and targ is
				// a generic function, we need to initialize u.y with
				// the respective type parameters of targ.
				if !u.unify(par.typ, targ) {
					errorf("type", par.typ, targ, arg)
					return nil
				}
			} else {
				indices = append(indices, i)
			}
		}
	}

	// If we've got all type arguments, we're done.
	var index int
	targs, index = u.x.types()
	if index < 0 {
		return targs
	}

	// See how far we get with constraint type inference.
	// Note that even if we don't have any type arguments, constraint type inference
	// may produce results for constraints that explicitly specify a type.
	targs, index = check.inferB(tparams, targs, report)
	if targs == nil || index < 0 {
		return targs
	}

	// --- 3 ---
	// Use any untyped arguments to infer additional type arguments.
	// Some generic parameters with untyped arguments may have been given
	// a type by now, we can ignore them.
	for _, i := range indices {
		par := params.At(i)
		// Since untyped types are all basic (i.e., non-composite) types, an
		// untyped argument will never match a composite parameter type; the
		// only parameter type it can possibly match against is a *TypeParam.
		// Thus, only consider untyped arguments for generic parameters that
		// are not of composite types and which don't have a type inferred yet.
		if tpar, _ := par.typ.(*_TypeParam); tpar != nil && targs[tpar.index] == nil {
			arg := args[i]
			targ := Default(arg.typ)
			// The default type for an untyped nil is untyped nil. We must not
			// infer an untyped nil type as type parameter type. Ignore untyped
			// nil by making sure all default argument types are typed.
			if isTyped(targ) && !u.unify(par.typ, targ) {
				errorf("default type", par.typ, targ, arg)
				return nil
			}
		}
	}

	// If we've got all type arguments, we're done.
	targs, index = u.x.types()
	if index < 0 {
		return targs
	}

	// Again, follow up with constraint type inference.
	targs, index = check.inferB(tparams, targs, report)
	if targs == nil || index < 0 {
		return targs
	}

	// At least one type argument couldn't be inferred.
	assert(index >= 0 && targs[index] == nil)
	tpar := tparams[index]
	if report {
		check.errorf(posn, _Todo, "cannot infer %s (%v) (%v)", tpar.name, tpar.pos, targs)
	}
	return nil
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

// IsParameterized reports whether typ contains any of the type parameters of tparams.
func isParameterized(tparams []*TypeName, typ Type) bool {
	w := tpWalker{
		seen:    make(map[Type]bool),
		tparams: tparams,
	}
	return w.isParameterized(typ)
}

type tpWalker struct {
	seen    map[Type]bool
	tparams []*TypeName
}

func (w *tpWalker) isParameterized(typ Type) (res bool) {
	// detect cycles
	if x, ok := w.seen[typ]; ok {
		return x
	}
	w.seen[typ] = false
	defer func() {
		w.seen[typ] = res
	}()

	switch t := typ.(type) {
	case nil, *Basic: // TODO(gri) should nil be handled here?
		break

	case *Array:
		return w.isParameterized(t.elem)

	case *Slice:
		return w.isParameterized(t.elem)

	case *Struct:
		for _, fld := range t.fields {
			if w.isParameterized(fld.typ) {
				return true
			}
		}

	case *Pointer:
		return w.isParameterized(t.base)

	case *Tuple:
		n := t.Len()
		for i := 0; i < n; i++ {
			if w.isParameterized(t.At(i).typ) {
				return true
			}
		}

	case *_Sum:
		return w.isParameterizedList(t.types)

	case *Signature:
		// t.tparams may not be nil if we are looking at a signature
		// of a generic function type (or an interface method) that is
		// part of the type we're testing. We don't care about these type
		// parameters.
		// Similarly, the receiver of a method may declare (rather then
		// use) type parameters, we don't care about those either.
		// Thus, we only need to look at the input and result parameters.
		return w.isParameterized(t.params) || w.isParameterized(t.results)

	case *Interface:
		if t.allMethods != nil {
			// TODO(rFindley) at some point we should enforce completeness here
			for _, m := range t.allMethods {
				if w.isParameterized(m.typ) {
					return true
				}
			}
			return w.isParameterizedList(unpackType(t.allTypes))
		}

		return t.iterate(func(t *Interface) bool {
			for _, m := range t.methods {
				if w.isParameterized(m.typ) {
					return true
				}
			}
			return w.isParameterizedList(unpackType(t.types))
		}, nil)

	case *Map:
		return w.isParameterized(t.key) || w.isParameterized(t.elem)

	case *Chan:
		return w.isParameterized(t.elem)

	case *Named:
		return w.isParameterizedList(t.targs)

	case *_TypeParam:
		// t must be one of w.tparams
		return t.index < len(w.tparams) && w.tparams[t.index].typ == t

	case *instance:
		return w.isParameterizedList(t.targs)

	default:
		unreachable()
	}

	return false
}

func (w *tpWalker) isParameterizedList(list []Type) bool {
	for _, t := range list {
		if w.isParameterized(t) {
			return true
		}
	}
	return false
}

// inferB returns the list of actual type arguments inferred from the type parameters'
// bounds and an initial set of type arguments. If type inference is impossible because
// unification fails, an error is reported if report is set to true, the resulting types
// list is nil, and index is 0.
// Otherwise, types is the list of inferred type arguments, and index is the index of the
// first type argument in that list that couldn't be inferred (and thus is nil). If all
// type arguments were inferred successfully, index is < 0. The number of type arguments
// provided may be less than the number of type parameters, but there must be at least one.
func (check *Checker) inferB(tparams []*TypeName, targs []Type, report bool) (types []Type, index int) {
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
		typ := tpar.typ.(*_TypeParam)
		sbound := check.structuralType(typ.bound)
		if sbound != nil {
			if !u.unify(typ, sbound) {
				if report {
					check.errorf(tpar, _Todo, "%s does not match %s", tpar, sbound)
				}
				return nil, 0
			}
		}
	}

	// u.x.types() now contains the incoming type arguments plus any additional type
	// arguments for which there were structural constraints. The newly inferred non-
	// nil entries may still contain references to other type parameters. For instance,
	// for [A any, B interface{type []C}, C interface{type *A}], if A == int
	// was given, unification produced the type list [int, []C, *A]. We eliminate the
	// remaining type parameters by substituting the type parameters in this type list
	// until nothing changes anymore.
	types, _ = u.x.types()
	if debug {
		for i, targ := range targs {
			assert(targ == nil || types[i] == targ)
		}
	}

	// dirty tracks the indices of all types that may still contain type parameters.
	// We know that nil type entries and entries corresponding to provided (non-nil)
	// type arguments are clean, so exclude them from the start.
	var dirty []int
	for i, typ := range types {
		if typ != nil && (i >= len(targs) || targs[i] == nil) {
			dirty = append(dirty, i)
		}
	}

	for len(dirty) > 0 {
		// TODO(gri) Instead of creating a new substMap for each iteration,
		// provide an update operation for substMaps and only change when
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

	// Once nothing changes anymore, we may still have type parameters left;
	// e.g., a structural constraint *P may match a type parameter Q but we
	// don't have any type arguments to fill in for *P or Q (issue #45548).
	// Don't let such inferences escape, instead nil them out.
	for i, typ := range types {
		if typ != nil && isParameterized(tparams, typ) {
			types[i] = nil
		}
	}

	// update index
	index = -1
	for i, typ := range types {
		if typ == nil {
			index = i
			break
		}
	}

	return
}

// structuralType returns the structural type of a constraint, if any.
func (check *Checker) structuralType(constraint Type) Type {
	if iface, _ := under(constraint).(*Interface); iface != nil {
		check.completeInterface(token.NoPos, iface)
		types := unpackType(iface.allTypes)
		if len(types) == 1 {
			return types[0]
		}
		return nil
	}
	return constraint
}
