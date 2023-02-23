// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type parameter inference.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// If compareWithInfer1, infer2 results must match infer1 results.
// Disable before releasing Go 1.21.
const compareWithInfer1 = false

// infer attempts to infer the complete set of type arguments for generic function instantiation/call
// based on the given type parameters tparams, type arguments targs, function parameters params, and
// function arguments args, if any. There must be at least one type parameter, no more type arguments
// than type parameters, and params and args must match in number (incl. zero).
// If successful, infer returns the complete list of given and inferred type arguments, one for each
// type parameter. Otherwise the result is nil and appropriate errors will be reported.
func (check *Checker) infer(pos syntax.Pos, tparams []*TypeParam, targs []Type, params *Tuple, args []*operand) []Type {
	r2 := check.infer2(pos, tparams, targs, params, args)

	if compareWithInfer1 {
		r1 := check.infer1(pos, tparams, targs, params, args, r2 == nil) // be silent on errors if infer2 failed
		assert(len(r2) == len(r1))
		for i, targ2 := range r2 {
			targ1 := r1[i]
			var c comparer
			c.ignoreInvalids = true
			if !c.identical(targ2, targ1, nil) {
				tpar := tparams[i]
				check.dump("%v: type argument for %s: infer1: %s, infer2: %s", tpar.Obj().Pos(), tpar, targ1, targ2)
				panic("inconsistent type inference")
			}
		}
	}

	return r2
}

// infer2 is an implementation of infer.
func (check *Checker) infer2(pos syntax.Pos, tparams []*TypeParam, targs []Type, params *Tuple, args []*operand) (inferred []Type) {
	if debug {
		defer func() {
			assert(inferred == nil || len(inferred) == len(tparams))
			for _, targ := range inferred {
				assert(targ != nil)
			}
		}()
	}

	if traceInference {
		check.dump("-- infer2 %s%s ➞ %s", tparams, params, targs)
		defer func() {
			check.dump("=> %s ➞ %s\n", tparams, inferred)
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

	// Rename type parameters to avoid conflicts in recursive instantiation scenarios.
	tparams, params = check.renameTParams(pos, tparams, params)

	if traceInference {
		check.dump("after rename: %s%s ➞ %s\n", tparams, params, targs)
	}

	// Make sure we have a "full" list of type arguments, some of which may
	// be nil (unknown). Make a copy so as to not clobber the incoming slice.
	if len(targs) < n {
		targs2 := make([]Type, n)
		copy(targs2, targs)
		targs = targs2
	}
	// len(targs) == n

	// Continue with the type arguments we have. Avoid matching generic
	// parameters that already have type arguments against function arguments:
	// It may fail because matching uses type identity while parameter passing
	// uses assignment rules. Instantiate the parameter list with the type
	// arguments we have, and continue with that parameter list.

	// Substitute type arguments for their respective type parameters in params,
	// if any. Note that nil targs entries are ignored by check.subst.
	// TODO(gri) Can we avoid this (we're setting known type arguments below,
	//           but that doesn't impact the isParameterized check for now).
	if params.Len() > 0 {
		smap := makeSubstMap(tparams, targs)
		params = check.subst(nopos, params, smap, nil, check.context()).(*Tuple)
	}

	// Unify parameter and argument types for generic parameters with typed arguments
	// and collect the indices of generic parameters with untyped arguments.
	// Terminology: generic parameter = function parameter with a type-parameterized type
	u := newUnifier(tparams, targs)

	errorf := func(kind string, tpar, targ Type, arg *operand) {
		// provide a better error message if we can
		targs := u.inferred(tparams)
		if targs[0] == nil {
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
				check.errorf(arg, CannotInferTypeArgs, "%s %s of %s does not match %s (cannot infer %s)", kind, targ, arg.expr, tpar, typeParamsString(tparams))
				return
			}
		}
		smap := makeSubstMap(tparams, targs)
		// TODO(gri): pass a poser here, rather than arg.Pos().
		inferred := check.subst(arg.Pos(), tpar, smap, nil, check.context())
		// CannotInferTypeArgs indicates a failure of inference, though the actual
		// error may be better attributed to a user-provided type argument (hence
		// InvalidTypeArg). We can't differentiate these cases, so fall back on
		// the more general CannotInferTypeArgs.
		if inferred != tpar {
			check.errorf(arg, CannotInferTypeArgs, "%s %s of %s does not match inferred type %s for %s", kind, targ, arg.expr, inferred, tpar)
		} else {
			check.errorf(arg, CannotInferTypeArgs, "%s %s of %s does not match %s", kind, targ, arg.expr, tpar)
		}
	}

	// indices of generic parameters with untyped arguments, for later use
	var untyped []int

	// --- 1 ---
	// use information from function arguments

	if traceInference {
		u.tracef("parameters: %s", params)
		u.tracef("arguments : %s", args)
	}

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
			if isTyped(arg.typ) {
				if !u.unify(par.typ, arg.typ) {
					errorf("type", par.typ, arg.typ, arg)
					return nil
				}
			} else if _, ok := par.typ.(*TypeParam); ok {
				// Since default types are all basic (i.e., non-composite) types, an
				// untyped argument will never match a composite parameter type; the
				// only parameter type it can possibly match against is a *TypeParam.
				// Thus, for untyped arguments we only need to look at parameter types
				// that are single type parameters.
				untyped = append(untyped, i)
			}
		}
	}

	if traceInference {
		inferred := u.inferred(tparams)
		u.tracef("=> %s ➞ %s\n", tparams, inferred)
	}

	// --- 2 ---
	// use information from type parameter constraints

	if traceInference {
		u.tracef("type parameters: %s", tparams)
	}

	// Repeatedly apply constraint type inference as long as
	// progress is being made.
	//
	// This is an O(n^2) algorithm where n is the number of
	// type parameters: if there is progress, at least one
	// type argument is inferred per iteration and we have
	// a doubly nested loop.
	//
	// In practice this is not a problem because the number
	// of type parameters tends to be very small (< 5 or so).
	// (It should be possible for unification to efficiently
	// signal newly inferred type arguments; then the loops
	// here could handle the respective type parameters only,
	// but that will come at a cost of extra complexity which
	// may not be worth it.)
	for {
		nn := u.unknowns()

		for _, tpar := range tparams {
			// If there is a core term (i.e., a core type with tilde information)
			// unify the type parameter with the core type.
			if core, single := coreTerm(tpar); core != nil {
				if traceInference {
					u.tracef("core(%s) = %s (single = %v)", tpar, core, single)
				}
				// A type parameter can be unified with its core type in two cases.
				tx := u.at(tpar)
				switch {
				case tx != nil:
					// The corresponding type argument tx is known.
					// In this case, if the core type has a tilde, the type argument's underlying
					// type must match the core type, otherwise the type argument and the core type
					// must match.
					// If tx is an (external) type parameter, don't consider its underlying type
					// (which is an interface). The unifier will use the type parameter's core
					// type automatically.
					if core.tilde && !isTypeParam(tx) {
						tx = under(tx)
					}
					if !u.unify(tx, core.typ) {
						check.errorf(pos, CannotInferTypeArgs, "%s does not match %s", tpar, core.typ)
						return nil
					}
				case single && !core.tilde:
					// The corresponding type argument tx is unknown and there's a single
					// specific type and no tilde.
					// In this case the type argument must be that single type; set it.
					u.set(tpar, core.typ)
				}
			} else {
				if traceInference {
					u.tracef("core(%s) = nil", tpar)
				}
			}
		}

		if u.unknowns() == nn {
			break // no progress
		}
	}

	if traceInference {
		inferred := u.inferred(tparams)
		u.tracef("=> %s ➞ %s\n", tparams, inferred)
	}

	// --- 3 ---
	// use information from untyped contants

	if traceInference {
		u.tracef("untyped: %v", untyped)
	}

	// Some generic parameters with untyped arguments may have been given a type by now.
	// Collect all remaining parameters that don't have a type yet and unify them with
	// the default types of the untyped arguments.
	// We need to collect them all before unifying them with their untyped arguments;
	// otherwise a parameter type that appears multiple times will have a type after
	// the first unification and will be skipped later on, leading to incorrect results.
	j := 0
	for _, i := range untyped {
		tpar := params.At(i).typ.(*TypeParam) // is type parameter by construction of untyped
		if u.at(tpar) == nil {
			untyped[j] = i
			j++
		}
	}
	// untyped[:j] are the undices of parameters without a type yet
	for _, i := range untyped[:j] {
		tpar := params.At(i).typ.(*TypeParam)
		arg := args[i]
		typ := Default(arg.typ)
		// The default type for an untyped nil is untyped nil which must
		// not be inferred as type parameter type. Ignore them by making
		// sure all default types are typed.
		if isTyped(typ) && !u.unify(tpar, typ) {
			errorf("default type", tpar, typ, arg)
			return nil
		}
	}

	// --- simplify ---

	// u.inferred(tparams) now contains the incoming type arguments plus any additional type
	// arguments which were inferred. The inferred non-nil entries may still contain
	// references to other type parameters found in constraints.
	// For instance, for [A any, B interface{ []C }, C interface{ *A }], if A == int
	// was given, unification produced the type list [int, []C, *A]. We eliminate the
	// remaining type parameters by substituting the type parameters in this type list
	// until nothing changes anymore.
	inferred = u.inferred(tparams)
	if debug {
		for i, targ := range targs {
			assert(targ == nil || inferred[i] == targ)
		}
	}

	// The data structure of each (provided or inferred) type represents a graph, where
	// each node corresponds to a type and each (directed) vertex points to a component
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
	// through a type parameter is detected, cycleFinder nils out the respective type
	// which kills the cycle; this also means that the respective type could not be
	// inferred.
	//
	// TODO(gri) If useful, we could report the respective cycle as an error. We don't
	//           do this now because type inference will fail anyway, and furthermore,
	//           constraints with cycles of this kind cannot currently be satisfied by
	//           any user-supplied type. But should that change, reporting an error
	//           would be wrong.
	w := cycleFinder{tparams, inferred, make(map[Type]bool)}
	for _, t := range tparams {
		w.typ(t) // t != nil
	}

	// dirty tracks the indices of all types that may still contain type parameters.
	// We know that nil type entries and entries corresponding to provided (non-nil)
	// type arguments are clean, so exclude them from the start.
	var dirty []int
	for i, typ := range inferred {
		if typ != nil && (i >= len(targs) || targs[i] == nil) {
			dirty = append(dirty, i)
		}
	}

	for len(dirty) > 0 {
		// TODO(gri) Instead of creating a new substMap for each iteration,
		// provide an update operation for substMaps and only change when
		// needed. Optimization.
		smap := makeSubstMap(tparams, inferred)
		n := 0
		for _, index := range dirty {
			t0 := inferred[index]
			if t1 := check.subst(nopos, t0, smap, nil, check.context()); t1 != t0 {
				inferred[index] = t1
				dirty[n] = index
				n++
			}
		}
		dirty = dirty[:n]
	}

	// Once nothing changes anymore, we may still have type parameters left;
	// e.g., a constraint with core type *P may match a type parameter Q but
	// we don't have any type arguments to fill in for *P or Q (go.dev/issue/45548).
	// Don't let such inferences escape; instead treat them as unresolved.
	for i, typ := range inferred {
		if typ == nil || isParameterized(tparams, typ) {
			obj := tparams[i].obj
			check.errorf(pos, CannotInferTypeArgs, "cannot infer %s (%s)", obj.name, obj.pos)
			return nil
		}
	}

	return
}

// dummy function using syntax.Pos to satisfy go/types generator for now
// TODO(gri) remove and adjust generator
func _(syntax.Pos) {}
