// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type parameter inference given
// a list of concrete arguments and a parameter list.

package types

import (
	"fmt"
	"go/token"
	"strings"
)

// infer attempts to infer the complete set of type arguments for generic function instantiation/call
// based on the given type parameters tparams, type arguments targs, function parameters params, and
// function arguments args, if any. There must be at least one type parameter, no more type arguments
// than type parameters, and params and args must match in number (incl. zero).
// If successful, infer returns the complete list of type arguments, one for each type parameter.
// Otherwise the result is nil and appropriate errors will be reported.
//
// Inference proceeds as follows:
//
//   Starting with given type arguments
//   1) apply FTI (function type inference) with typed arguments,
//   2) apply CTI (constraint type inference),
//   3) apply FTI with untyped function arguments,
//   4) apply CTI.
//
// The process stops as soon as all type arguments are known or an error occurs.
func (check *Checker) infer(posn positioner, tparams []*TypeParam, targs []Type, params *Tuple, args []*operand) (result []Type) {
	if debug {
		defer func() {
			assert(result == nil || len(result) == len(tparams))
			for _, targ := range result {
				assert(targ != nil)
			}
			//check.dump("### inferred targs = %s", result)
		}()
	}

	if traceInference {
		check.dump("-- inferA %s%s ➞ %s", tparams, params, targs)
		defer func() {
			check.dump("=> inferA %s ➞ %s", tparams, result)
		}()
	}

	// There must be at least one type parameter, and no more type arguments than type parameters.
	n := len(tparams)
	assert(n > 0 && len(targs) <= n)

	// Function parameters and arguments must match in number.
	assert(params.Len() == len(args))

	// If we already have all type arguments, we're done.
	if len(targs) == n {
		return targs
	}
	// len(targs) < n

	const enableTparamRenaming = true
	if enableTparamRenaming {
		// For the purpose of type inference we must differentiate type parameters
		// occurring in explicit type or value function arguments from the type
		// parameters we are solving for via unification, because they may be the
		// same in self-recursive calls. For example:
		//
		//  func f[P *Q, Q any](p P, q Q) {
		//    f(p)
		//  }
		//
		// In this example, the fact that the P used in the instantation f[P] has
		// the same pointer identity as the P we are trying to solve for via
		// unification is coincidental: there is nothing special about recursive
		// calls that should cause them to conflate the identity of type arguments
		// with type parameters. To put it another way: any such self-recursive
		// call is equivalent to a mutually recursive call, which does not run into
		// any problems of type parameter identity. For example, the following code
		// is equivalent to the code above.
		//
		//  func f[P interface{*Q}, Q any](p P, q Q) {
		//    f2(p)
		//  }
		//
		//  func f2[P interface{*Q}, Q any](p P, q Q) {
		//    f(p)
		//  }
		//
		// We can turn the first example into the second example by renaming type
		// parameters in the original signature to give them a new identity. As an
		// optimization, we do this only for self-recursive calls.

		// We can detect if we are in a self-recursive call by comparing the
		// identity of the first type parameter in the current function with the
		// first type parameter in tparams. This works because type parameters are
		// unique to their type parameter list.
		selfRecursive := check.sig != nil && check.sig.tparams.Len() > 0 && tparams[0] == check.sig.tparams.At(0)

		if selfRecursive {
			// In self-recursive inference, rename the type parameters with new type
			// parameters that are the same but for their pointer identity.
			tparams2 := make([]*TypeParam, len(tparams))
			for i, tparam := range tparams {
				tname := NewTypeName(tparam.Obj().Pos(), tparam.Obj().Pkg(), tparam.Obj().Name(), nil)
				tparams2[i] = NewTypeParam(tname, nil)
				tparams2[i].index = tparam.index // == i
			}

			renameMap := makeRenameMap(tparams, tparams2)
			for i, tparam := range tparams {
				tparams2[i].bound = check.subst(posn.Pos(), tparam.bound, renameMap, nil)
			}

			tparams = tparams2
			params = check.subst(posn.Pos(), params, renameMap, nil).(*Tuple)
		}
	}

	// If we have more than 2 arguments, we may have arguments with named and unnamed types.
	// If that is the case, permutate params and args such that the arguments with named
	// types are first in the list. This doesn't affect type inference if all types are taken
	// as is. But when we have inexact unification enabled (as is the case for function type
	// inference), when a named type is unified with an unnamed type, unification proceeds
	// with the underlying type of the named type because otherwise unification would fail
	// right away. This leads to an asymmetry in type inference: in cases where arguments of
	// named and unnamed types are passed to parameters with identical type, different types
	// (named vs underlying) may be inferred depending on the order of the arguments.
	// By ensuring that named types are seen first, order dependence is avoided and unification
	// succeeds where it can.
	//
	// This code is disabled for now pending decision whether we want to address cases like
	// these and make the spec on type inference more complicated (see issue #43056).
	const enableArgSorting = false
	if m := len(args); m >= 2 && enableArgSorting {
		// Determine indices of arguments with named and unnamed types.
		var named, unnamed []int
		for i, arg := range args {
			if hasName(arg.typ) {
				named = append(named, i)
			} else {
				unnamed = append(unnamed, i)
			}
		}

		// If we have named and unnamed types, move the arguments with
		// named types first. Update the parameter list accordingly.
		// Make copies so as not to clobber the incoming slices.
		if len(named) != 0 && len(unnamed) != 0 {
			params2 := make([]*Var, m)
			args2 := make([]*operand, m)
			i := 0
			for _, j := range named {
				params2[i] = params.At(j)
				args2[i] = args[j]
				i++
			}
			for _, j := range unnamed {
				params2[i] = params.At(j)
				args2[i] = args[j]
				i++
			}
			params = NewTuple(params2...)
			args = args2
		}
	}

	// --- 1 ---
	// Continue with the type arguments we have. Avoid matching generic
	// parameters that already have type arguments against function arguments:
	// It may fail because matching uses type identity while parameter passing
	// uses assignment rules. Instantiate the parameter list with the type
	// arguments we have, and continue with that parameter list.

	// First, make sure we have a "full" list of type arguments, some of which
	// may be nil (unknown). Make a copy so as to not clobber the incoming slice.
	if len(targs) < n {
		targs2 := make([]Type, n)
		copy(targs2, targs)
		targs = targs2
	}
	// len(targs) == n

	// Substitute type arguments for their respective type parameters in params,
	// if any. Note that nil targs entries are ignored by check.subst.
	// TODO(gri) Can we avoid this (we're setting known type arguments below,
	//           but that doesn't impact the isParameterized check for now).
	if params.Len() > 0 {
		smap := makeSubstMap(tparams, targs)
		params = check.subst(token.NoPos, params, smap, nil).(*Tuple)
	}

	// Unify parameter and argument types for generic parameters with typed arguments
	// and collect the indices of generic parameters with untyped arguments.
	// Terminology: generic parameter = function parameter with a type-parameterized type
	u := newUnifier(false)
	u.x.init(tparams)

	// Set the type arguments which we know already.
	for i, targ := range targs {
		if targ != nil {
			u.x.set(i, targ)
		}
	}

	errorf := func(kind string, tpar, targ Type, arg *operand) {
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
				check.errorf(arg, _CannotInferTypeArgs, "%s %s of %s does not match %s (cannot infer %s)", kind, targ, arg.expr, tpar, typeParamsString(tparams))
				return
			}
		}
		smap := makeSubstMap(tparams, targs)
		// TODO(rFindley): pass a positioner here, rather than arg.Pos().
		inferred := check.subst(arg.Pos(), tpar, smap, nil)
		// _CannotInferTypeArgs indicates a failure of inference, though the actual
		// error may be better attributed to a user-provided type argument (hence
		// _InvalidTypeArg). We can't differentiate these cases, so fall back on
		// the more general _CannotInferTypeArgs.
		if inferred != tpar {
			check.errorf(arg, _CannotInferTypeArgs, "%s %s of %s does not match inferred type %s for %s", kind, targ, arg.expr, inferred, tpar)
		} else {
			check.errorf(arg, _CannotInferTypeArgs, "%s %s of %s does not match %s", kind, targ, arg.expr, tpar)
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
				// targs resulting in fewer follow-on errors.
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
			} else if _, ok := par.typ.(*TypeParam); ok {
				// Since default types are all basic (i.e., non-composite) types, an
				// untyped argument will never match a composite parameter type; the
				// only parameter type it can possibly match against is a *TypeParam.
				// Thus, for untyped arguments we only need to look at parameter types
				// that are single type parameters.
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

	// --- 2 ---
	// See how far we get with constraint type inference.
	// Note that even if we don't have any type arguments, constraint type inference
	// may produce results for constraints that explicitly specify a type.
	targs, index = check.inferB(posn, tparams, targs)
	if targs == nil || index < 0 {
		return targs
	}

	// --- 3 ---
	// Use any untyped arguments to infer additional type arguments.
	// Some generic parameters with untyped arguments may have been given
	// a type by now, we can ignore them.
	for _, i := range indices {
		tpar := params.At(i).typ.(*TypeParam) // is type parameter by construction of indices
		// Only consider untyped arguments for which the corresponding type
		// parameter doesn't have an inferred type yet.
		if targs[tpar.index] == nil {
			arg := args[i]
			targ := Default(arg.typ)
			// The default type for an untyped nil is untyped nil. We must not
			// infer an untyped nil type as type parameter type. Ignore untyped
			// nil by making sure all default argument types are typed.
			if isTyped(targ) && !u.unify(tpar, targ) {
				errorf("default type", tpar, targ, arg)
				return nil
			}
		}
	}

	// If we've got all type arguments, we're done.
	targs, index = u.x.types()
	if index < 0 {
		return targs
	}

	// --- 4 ---
	// Again, follow up with constraint type inference.
	targs, index = check.inferB(posn, tparams, targs)
	if targs == nil || index < 0 {
		return targs
	}

	// At least one type argument couldn't be inferred.
	assert(index >= 0 && targs[index] == nil)
	tpar := tparams[index]
	check.errorf(posn, _CannotInferTypeArgs, "cannot infer %s (%v)", tpar.obj.name, tpar.obj.pos)
	return nil
}

// typeParamsString produces a string containing all the type parameter names
// in list suitable for human consumption.
func typeParamsString(list []*TypeParam) string {
	// common cases
	n := len(list)
	switch n {
	case 0:
		return ""
	case 1:
		return list[0].obj.name
	case 2:
		return list[0].obj.name + " and " + list[1].obj.name
	}

	// general case (n > 2)
	var b strings.Builder
	for i, tname := range list[:n-1] {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(tname.obj.name)
	}
	b.WriteString(", and ")
	b.WriteString(list[n-1].obj.name)
	return b.String()
}

// IsParameterized reports whether typ contains any of the type parameters of tparams.
func isParameterized(tparams []*TypeParam, typ Type) bool {
	w := tpWalker{
		seen:    make(map[Type]bool),
		tparams: tparams,
	}
	return w.isParameterized(typ)
}

type tpWalker struct {
	seen    map[Type]bool
	tparams []*TypeParam
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
		tset := t.typeSet()
		for _, m := range tset.methods {
			if w.isParameterized(m.typ) {
				return true
			}
		}
		return tset.is(func(t *term) bool {
			return t != nil && w.isParameterized(t.typ)
		})

	case *Map:
		return w.isParameterized(t.key) || w.isParameterized(t.elem)

	case *Chan:
		return w.isParameterized(t.elem)

	case *Named:
		return w.isParameterizedTypeList(t.targs.list())

	case *TypeParam:
		// t must be one of w.tparams
		return tparamIndex(w.tparams, t) >= 0

	default:
		unreachable()
	}

	return false
}

func (w *tpWalker) isParameterizedTypeList(list []Type) bool {
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
func (check *Checker) inferB(posn positioner, tparams []*TypeParam, targs []Type) (types []Type, index int) {
	assert(len(tparams) >= len(targs) && len(targs) > 0)

	if traceInference {
		check.dump("-- inferB %s ➞ %s", tparams, targs)
		defer func() {
			check.dump("=> inferB %s ➞ %s", tparams, types)
		}()
	}

	// Setup bidirectional unification between constraints
	// and the corresponding type arguments (which may be nil!).
	u := newUnifier(false)
	u.x.init(tparams)
	u.y = u.x // type parameters between LHS and RHS of unification are identical

	// Set the type arguments which we know already.
	for i, targ := range targs {
		if targ != nil {
			u.x.set(i, targ)
		}
	}

	// Repeatedly apply constraint type inference as long as
	// there are still unknown type arguments and progress is
	// being made.
	//
	// This is an O(n^2) algorithm where n is the number of
	// type parameters: if there is progress (and iteration
	// continues), at least one type argument is inferred
	// per iteration and we have a doubly nested loop.
	// In practice this is not a problem because the number
	// of type parameters tends to be very small (< 5 or so).
	// (It should be possible for unification to efficiently
	// signal newly inferred type arguments; then the loops
	// here could handle the respective type parameters only,
	// but that will come at a cost of extra complexity which
	// may not be worth it.)
	for n := u.x.unknowns(); n > 0; {
		nn := n

		for i, tpar := range tparams {
			// If there is a core term (i.e., a core type with tilde information)
			// unify the type parameter with the core type.
			if core, single := coreTerm(tpar); core != nil {
				// A type parameter can be unified with its core type in two cases.
				tx := u.x.at(i)
				switch {
				case tx != nil:
					// The corresponding type argument tx is known.
					// In this case, if the core type has a tilde, the type argument's underlying
					// type must match the core type, otherwise the type argument and the core type
					// must match.
					// If tx is an external type parameter, don't consider its underlying type
					// (which is an interface). Core type unification will attempt to unify against
					// core.typ.
					// Note also that even with inexact unification we cannot leave away the under
					// call here because it's possible that both tx and core.typ are named types,
					// with under(tx) being a (named) basic type matching core.typ. Such cases do
					// not match with inexact unification.
					if core.tilde && !isTypeParam(tx) {
						tx = under(tx)
					}
					if !u.unify(tx, core.typ) {
						// TODO(gri) improve error message by providing the type arguments
						//           which we know already
						// Don't use term.String() as it always qualifies types, even if they
						// are in the current package.
						tilde := ""
						if core.tilde {
							tilde = "~"
						}
						check.errorf(posn, _InvalidTypeArg, "%s does not match %s%s", tpar, tilde, core.typ)
						return nil, 0
					}

				case single && !core.tilde:
					// The corresponding type argument tx is unknown and there's a single
					// specific type and no tilde.
					// In this case the type argument must be that single type; set it.
					u.x.set(i, core.typ)

				default:
					// Unification is not possible and no progress was made.
					continue
				}

				// The number of known type arguments may have changed.
				nn = u.x.unknowns()
				if nn == 0 {
					break // all type arguments are known
				}
			}
		}

		assert(nn <= n)
		if nn == n {
			break // no progress
		}
		n = nn
	}

	// u.x.types() now contains the incoming type arguments plus any additional type
	// arguments which were inferred from core terms. The newly inferred non-nil
	// entries may still contain references to other type parameters.
	// For instance, for [A any, B interface{ []C }, C interface{ *A }], if A == int
	// was given, unification produced the type list [int, []C, *A]. We eliminate the
	// remaining type parameters by substituting the type parameters in this type list
	// until nothing changes anymore.
	types, _ = u.x.types()
	if debug {
		for i, targ := range targs {
			assert(targ == nil || types[i] == targ)
		}
	}

	// The data structure of each (provided or inferred) type represents a graph, where
	// each node corresponds to a type and each (directed) vertice points to a component
	// type. The substitution process described above repeatedly replaces type parameter
	// nodes in these graphs with the graphs of the types the type parameters stand for,
	// which creates a new (possibly bigger) graph for each type.
	// The substitution process will not stop if the replacement graph for a type parameter
	// also contains that type parameter.
	// For instance, for [A interface{ *A }], without any type argument provided for A,
	// unification produces the type list [*A]. Substituting A in *A with the value for
	// A will lead to infinite expansion by producing [**A], [****A], [********A], etc.,
	// because the graph A -> *A has a cycle through A.
	// Generally, cycles may occur across multiple type parameters and inferred types
	// (for instance, consider [P interface{ *Q }, Q interface{ func(P) }]).
	// We eliminate cycles by walking the graphs for all type parameters. If a cycle
	// through a type parameter is detected, cycleFinder nils out the respectice type
	// which kills the cycle; this also means that the respective type could not be
	// inferred.
	//
	// TODO(gri) If useful, we could report the respective cycle as an error. We don't
	//           do this now because type inference will fail anyway, and furthermore,
	//           constraints with cycles of this kind cannot currently be satisfied by
	//           any user-suplied type. But should that change, reporting an error
	//           would be wrong.
	w := cycleFinder{tparams, types, make(map[Type]bool)}
	for _, t := range tparams {
		w.typ(t) // t != nil
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
			if t1 := check.subst(token.NoPos, t0, smap, nil); t1 != t0 {
				types[index] = t1
				dirty[n] = index
				n++
			}
		}
		dirty = dirty[:n]
	}

	// Once nothing changes anymore, we may still have type parameters left;
	// e.g., a constraint with core type *P may match a type parameter Q but
	// we don't have any type arguments to fill in for *P or Q (issue #45548).
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

// If the type parameter has a single specific type S, coreTerm returns (S, true).
// Otherwise, if tpar has a core type T, it returns a term corresponding to that
// core type and false. In that case, if any term of tpar has a tilde, the core
// term has a tilde. In all other cases coreTerm returns (nil, false).
func coreTerm(tpar *TypeParam) (*term, bool) {
	n := 0
	var single *term // valid if n == 1
	var tilde bool
	tpar.is(func(t *term) bool {
		if t == nil {
			assert(n == 0)
			return false // no terms
		}
		n++
		single = t
		if t.tilde {
			tilde = true
		}
		return true
	})
	if n == 1 {
		if debug {
			assert(debug && under(single.typ) == coreType(tpar))
		}
		return single, true
	}
	if typ := coreType(tpar); typ != nil {
		// A core type is always an underlying type.
		// If any term of tpar has a tilde, we don't
		// have a precise core type and we must return
		// a tilde as well.
		return &term{tilde, typ}, false
	}
	return nil, false
}

type cycleFinder struct {
	tparams []*TypeParam
	types   []Type
	seen    map[Type]bool
}

func (w *cycleFinder) typ(typ Type) {
	if w.seen[typ] {
		// We have seen typ before. If it is one of the type parameters
		// in tparams, iterative substitution will lead to infinite expansion.
		// Nil out the corresponding type which effectively kills the cycle.
		if tpar, _ := typ.(*TypeParam); tpar != nil {
			if i := tparamIndex(w.tparams, tpar); i >= 0 {
				// cycle through tpar
				w.types[i] = nil
			}
		}
		// If we don't have one of our type parameters, the cycle is due
		// to an ordinary recursive type and we can just stop walking it.
		return
	}
	w.seen[typ] = true
	defer delete(w.seen, typ)

	switch t := typ.(type) {
	case *Basic:
		// nothing to do

	case *Array:
		w.typ(t.elem)

	case *Slice:
		w.typ(t.elem)

	case *Struct:
		w.varList(t.fields)

	case *Pointer:
		w.typ(t.base)

	// case *Tuple:
	//      This case should not occur because tuples only appear
	//      in signatures where they are handled explicitly.

	case *Signature:
		if t.params != nil {
			w.varList(t.params.vars)
		}
		if t.results != nil {
			w.varList(t.results.vars)
		}

	case *Union:
		for _, t := range t.terms {
			w.typ(t.typ)
		}

	case *Interface:
		for _, m := range t.methods {
			w.typ(m.typ)
		}
		for _, t := range t.embeddeds {
			w.typ(t)
		}

	case *Map:
		w.typ(t.key)
		w.typ(t.elem)

	case *Chan:
		w.typ(t.elem)

	case *Named:
		for _, tpar := range t.TypeArgs().list() {
			w.typ(tpar)
		}

	case *TypeParam:
		if i := tparamIndex(w.tparams, t); i >= 0 && w.types[i] != nil {
			w.typ(w.types[i])
		}

	default:
		panic(fmt.Sprintf("unexpected %T", typ))
	}
}

func (w *cycleFinder) varList(list []*Var) {
	for _, v := range list {
		w.typ(v.typ)
	}
}
