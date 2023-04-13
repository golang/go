// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type parameter inference.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	. "internal/types/errors"
	"strings"
)

// infer attempts to infer the complete set of type arguments for generic function instantiation/call
// based on the given type parameters tparams, type arguments targs, function parameters params, and
// function arguments args, if any. There must be at least one type parameter, no more type arguments
// than type parameters, and params and args must match in number (incl. zero).
// If successful, infer returns the complete list of given and inferred type arguments, one for each
// type parameter. Otherwise the result is nil and appropriate errors will be reported.
func (check *Checker) infer(pos syntax.Pos, tparams []*TypeParam, targs []Type, params *Tuple, args []*operand) (inferred []Type) {
	if debug {
		defer func() {
			assert(inferred == nil || len(inferred) == len(tparams))
			for _, targ := range inferred {
				assert(targ != nil)
			}
		}()
	}

	if traceInference {
		check.dump("== infer : %s%s ➞ %s", tparams, params, targs) // aligned with rename print below
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
		check.dump("-- rename: %s%s ➞ %s\n", tparams, params, targs)
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
	// We do this for better error messages; it's not needed for correctness.
	// For instance, given:
	//
	//   func f[P, Q any](P, Q) {}
	//
	//   func _(s string) {
	//           f[int](s, s) // ERROR
	//   }
	//
	// With substitution, we get the error:
	//   "cannot use s (variable of type string) as int value in argument to f[int]"
	//
	// Without substitution we get the (worse) error:
	//   "type string of s does not match inferred type int for P"
	// even though the type int was provided (not inferred) for P.
	//
	// TODO(gri) We might be able to finesse this in the error message reporting
	//           (which only happens in case of an error) and then avoid doing
	//           the substitution (which always happens).
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
		u.tracef("== function parameters: %s", params)
		u.tracef("-- function arguments : %s", args)
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
		u.tracef("== type parameters: %s", tparams)
	}

	// Unify type parameters with their constraints as long
	// as progress is being made.
	//
	// This is an O(n^2) algorithm where n is the number of
	// type parameters: if there is progress, at least one
	// type argument is inferred per iteration, and we have
	// a doubly nested loop.
	//
	// In practice this is not a problem because the number
	// of type parameters tends to be very small (< 5 or so).
	// (It should be possible for unification to efficiently
	// signal newly inferred type arguments; then the loops
	// here could handle the respective type parameters only,
	// but that will come at a cost of extra complexity which
	// may not be worth it.)
	for i := 0; ; i++ {
		nn := u.unknowns()
		if traceInference {
			if i > 0 {
				fmt.Println()
			}
			u.tracef("-- iteration %d", i)
		}

		for _, tpar := range tparams {
			tx := u.at(tpar)
			core, single := coreTerm(tpar)
			if traceInference {
				u.tracef("-- type parameter %s = %s: core(%s) = %s, single = %v", tpar, tx, tpar, core, single)
			}

			// If there is a core term (i.e., a core type with tilde information)
			// unify the type parameter with the core type.
			if core != nil {
				// A type parameter can be unified with its core type in two cases.
				switch {
				case tx != nil:
					// The corresponding type argument tx is known. There are 2 cases:
					// 1) If the core type has a tilde, per spec requirement for tilde
					//    elements, the core type is an underlying (literal) type.
					//    And because of the tilde, the underlying type of tx must match
					//    against the core type.
					//    But because unify automatically matches a defined type against
					//    an underlying literal type, we can simply unify tx with the
					//    core type.
					// 2) If the core type doesn't have a tilde, we also must unify tx
					//    with the core type.
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
				if tx != nil {
					// We don't have a core type, but the type argument tx is known.
					// It must have (at least) all the methods of the type constraint,
					// and the method signatures must unify; otherwise tx cannot satisfy
					// the constraint.
					var cause string
					constraint := tpar.iface()
					if m, _ := check.missingMethod(tx, constraint, true, u.unify, &cause); m != nil {
						check.errorf(pos, CannotInferTypeArgs, "%s does not satisfy %s %s", tx, constraint, cause)
						return nil
					}
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
		u.tracef("== untyped arguments: %v", untyped)
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
	// untyped[:j] are the indices of parameters without a type yet
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
	// through a type parameter is detected, killCycles nils out the respective type
	// (in the inferred list) which kills the cycle, and marks the corresponding type
	// parameter as not inferred.
	//
	// TODO(gri) If useful, we could report the respective cycle as an error. We don't
	//           do this now because type inference will fail anyway, and furthermore,
	//           constraints with cycles of this kind cannot currently be satisfied by
	//           any user-supplied type. But should that change, reporting an error
	//           would be wrong.
	killCycles(tparams, inferred)

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

// renameTParams renames the type parameters in a function signature described by its
// type and ordinary parameters (tparams and params) such that each type parameter is
// given a new identity. renameTParams returns the new type and ordinary parameters.
func (check *Checker) renameTParams(pos syntax.Pos, tparams []*TypeParam, params *Tuple) ([]*TypeParam, *Tuple) {
	// For the purpose of type inference we must differentiate type parameters
	// occurring in explicit type or value function arguments from the type
	// parameters we are solving for via unification because they may be the
	// same in self-recursive calls:
	//
	//   func f[P constraint](x P) {
	//           f(x)
	//   }
	//
	// In this example, without type parameter renaming, the P used in the
	// instantiation f[P] has the same pointer identity as the P we are trying
	// to solve for through type inference. This causes problems for type
	// unification. Because any such self-recursive call is equivalent to
	// a mutually recursive call, type parameter renaming can be used to
	// create separate, disentangled type parameters. The above example
	// can be rewritten into the following equivalent code:
	//
	//   func f[P constraint](x P) {
	//           f2(x)
	//   }
	//
	//   func f2[P2 constraint](x P2) {
	//           f(x)
	//   }
	//
	// Type parameter renaming turns the first example into the second
	// example by renaming the type parameter P into P2.
	tparams2 := make([]*TypeParam, len(tparams))
	for i, tparam := range tparams {
		tname := NewTypeName(tparam.Obj().Pos(), tparam.Obj().Pkg(), tparam.Obj().Name(), nil)
		tparams2[i] = NewTypeParam(tname, nil)
		tparams2[i].index = tparam.index // == i
	}

	renameMap := makeRenameMap(tparams, tparams2)
	for i, tparam := range tparams {
		tparams2[i].bound = check.subst(pos, tparam.bound, renameMap, nil, check.context())
	}

	return tparams2, check.subst(pos, params, renameMap, nil, check.context()).(*Tuple)
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
	var buf strings.Builder
	for i, tname := range list[:n-1] {
		if i > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(tname.obj.name)
	}
	buf.WriteString(", and ")
	buf.WriteString(list[n-1].obj.name)
	return buf.String()
}

// isParameterized reports whether typ contains any of the type parameters of tparams.
func isParameterized(tparams []*TypeParam, typ Type) bool {
	w := tpWalker{
		tparams: tparams,
		seen:    make(map[Type]bool),
	}
	return w.isParameterized(typ)
}

type tpWalker struct {
	tparams []*TypeParam
	seen    map[Type]bool
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
	case *Basic:
		// nothing to do

	case *Array:
		return w.isParameterized(t.elem)

	case *Slice:
		return w.isParameterized(t.elem)

	case *Struct:
		return w.varList(t.fields)

	case *Pointer:
		return w.isParameterized(t.base)

	// case *Tuple:
	//      This case should not occur because tuples only appear
	//      in signatures where they are handled explicitly.

	case *Signature:
		// t.tparams may not be nil if we are looking at a signature
		// of a generic function type (or an interface method) that is
		// part of the type we're testing. We don't care about these type
		// parameters.
		// Similarly, the receiver of a method may declare (rather then
		// use) type parameters, we don't care about those either.
		// Thus, we only need to look at the input and result parameters.
		return t.params != nil && w.varList(t.params.vars) || t.results != nil && w.varList(t.results.vars)

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
		for _, t := range t.TypeArgs().list() {
			if w.isParameterized(t) {
				return true
			}
		}

	case *TypeParam:
		// t must be one of w.tparams
		return tparamIndex(w.tparams, t) >= 0

	default:
		panic(fmt.Sprintf("unexpected %T", typ))
	}

	return false
}

func (w *tpWalker) varList(list []*Var) bool {
	for _, v := range list {
		if w.isParameterized(v.typ) {
			return true
		}
	}
	return false
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

// killCycles walks through the given type parameters and looks for cycles
// created by type parameters whose inferred types refer back to that type
// parameter, either directly or indirectly. If such a cycle is detected,
// it is killed by setting the corresponding inferred type to nil.
//
// TODO(gri) Determine if we can simply abort inference as soon as we have
// found a single cycle.
func killCycles(tparams []*TypeParam, inferred []Type) {
	w := cycleFinder{tparams, inferred, make(map[Type]bool)}
	for _, t := range tparams {
		w.typ(t) // t != nil
	}
}

type cycleFinder struct {
	tparams  []*TypeParam
	inferred []Type
	seen     map[Type]bool
}

func (w *cycleFinder) typ(typ Type) {
	if w.seen[typ] {
		// We have seen typ before. If it is one of the type parameters
		// in w.tparams, iterative substitution will lead to infinite expansion.
		// Nil out the corresponding type which effectively kills the cycle.
		if tpar, _ := typ.(*TypeParam); tpar != nil {
			if i := tparamIndex(w.tparams, tpar); i >= 0 {
				// cycle through tpar
				w.inferred[i] = nil
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
		if i := tparamIndex(w.tparams, t); i >= 0 && w.inferred[i] != nil {
			w.typ(w.inferred[i])
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

// If tpar is a type parameter in list, tparamIndex returns the index
// of the type parameter in list. Otherwise the result is < 0.
func tparamIndex(list []*TypeParam, tpar *TypeParam) int {
	for i, p := range list {
		if p == tpar {
			return i
		}
	}
	return -1
}
