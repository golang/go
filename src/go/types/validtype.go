// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// validType verifies that the given type does not "expand" indefinitely
// producing a cycle in the type graph.
// (Cycles involving alias types, as in "type A = [10]A" are detected
// earlier, via the objDecl cycle detection mechanism.)
func (check *Checker) validType(typ *Named) {
	check.validType0(typ, nil, nil, nil)
}

// validType0 checks if the given type is valid. If typ is a type parameter
// its value is looked up in the provided environment. The environment is
// nil if typ is not part of (the RHS of) an instantiated type, in that case
// any type parameter encountered must be from an enclosing function and can
// be ignored. The nest list describes the stack (the "nest in memory") of
// types which contain (or embed in the case of interfaces) other types. For
// instance, a struct named S which contains a field of named type F contains
// (the memory of) F in S, leading to the nest S->F. If a type appears in its
// own nest (say S->F->S) we have an invalid recursive type. The path list is
// the full path of named types in a cycle, it is only needed for error reporting.
func (check *Checker) validType0(typ Type, env *tparamEnv, nest, path []*Named) bool {
	switch t := typ.(type) {
	case nil:
		// We should never see a nil type but be conservative and panic
		// only in debug mode.
		if debug {
			panic("validType0(nil)")
		}

	case *Array:
		return check.validType0(t.elem, env, nest, path)

	case *Struct:
		for _, f := range t.fields {
			if !check.validType0(f.typ, env, nest, path) {
				return false
			}
		}

	case *Union:
		for _, t := range t.terms {
			if !check.validType0(t.typ, env, nest, path) {
				return false
			}
		}

	case *Interface:
		for _, etyp := range t.embeddeds {
			if !check.validType0(etyp, env, nest, path) {
				return false
			}
		}

	case *Named:
		// Exit early if we already know t is valid.
		// This is purely an optimization but it prevents excessive computation
		// times in pathological cases such as testdata/fixedbugs/issue6977.go.
		// (Note: The valids map could also be allocated locally, once for each
		// validType call.)
		if check.valids.lookup(t) != nil {
			break
		}

		// Don't report a 2nd error if we already know the type is invalid
		// (e.g., if a cycle was detected earlier, via under).
		// Note: ensure that t.orig is fully resolved by calling Underlying().
		if t.Underlying() == Typ[Invalid] {
			return false
		}

		// If the current type t is also found in nest, (the memory of) t is
		// embedded in itself, indicating an invalid recursive type.
		for _, e := range nest {
			if Identical(e, t) {
				// t cannot be in an imported package otherwise that package
				// would have reported a type cycle and couldn't have been
				// imported in the first place.
				assert(t.obj.pkg == check.pkg)
				t.underlying = Typ[Invalid] // t is in the current package (no race possibility)
				// Find the starting point of the cycle and report it.
				// Because each type in nest must also appear in path (see invariant below),
				// type t must be in path since it was found in nest. But not every type in path
				// is in nest. Specifically t may appear in path with an earlier index than the
				// index of t in nest. Search again.
				for start, p := range path {
					if Identical(p, t) {
						check.cycleError(makeObjList(path[start:]))
						return false
					}
				}
				panic("cycle start not found")
			}
		}

		// No cycle was found. Check the RHS of t.
		// Every type added to nest is also added to path; thus every type that is in nest
		// must also be in path (invariant). But not every type in path is in nest, since
		// nest may be pruned (see below, *TypeParam case).
		if !check.validType0(t.Origin().fromRHS, env.push(t), append(nest, t), append(path, t)) {
			return false
		}

		check.valids.add(t) // t is valid

	case *TypeParam:
		// A type parameter stands for the type (argument) it was instantiated with.
		// Check the corresponding type argument for validity if we have one.
		if env != nil {
			if targ := env.tmap[t]; targ != nil {
				// Type arguments found in targ must be looked
				// up in the enclosing environment env.link. The
				// type argument must be valid in the enclosing
				// type (where the current type was instantiated),
				// hence we must check targ's validity in the type
				// nest excluding the current (instantiated) type
				// (see the example at the end of this file).
				// For error reporting we keep the full path.
				return check.validType0(targ, env.link, nest[:len(nest)-1], path)
			}
		}
	}

	return true
}

// makeObjList returns the list of type name objects for the given
// list of named types.
func makeObjList(tlist []*Named) []Object {
	olist := make([]Object, len(tlist))
	for i, t := range tlist {
		olist[i] = t.obj
	}
	return olist
}

// A tparamEnv provides the environment for looking up the type arguments
// with which type parameters for a given instance were instantiated.
// If we don't have an instance, the corresponding tparamEnv is nil.
type tparamEnv struct {
	tmap substMap
	link *tparamEnv
}

func (env *tparamEnv) push(typ *Named) *tparamEnv {
	// If typ is not an instantiated type there are no typ-specific
	// type parameters to look up and we don't need an environment.
	targs := typ.TypeArgs()
	if targs == nil {
		return nil // no instance => nil environment
	}

	// Populate tmap: remember the type argument for each type parameter.
	// We cannot use makeSubstMap because the number of type parameters
	// and arguments may not match due to errors in the source (too many
	// or too few type arguments). Populate tmap "manually".
	tparams := typ.TypeParams()
	n, m := targs.Len(), tparams.Len()
	if n > m {
		n = m // too many targs
	}
	tmap := make(substMap, n)
	for i := 0; i < n; i++ {
		tmap[tparams.At(i)] = targs.At(i)
	}

	return &tparamEnv{tmap: tmap, link: env}
}

// TODO(gri) Alternative implementation:
// We may not need to build a stack of environments to
// look up the type arguments for type parameters. The
// same information should be available via the path:
// We should be able to just walk the path backwards
// and find the type arguments in the instance objects.

// Here is an example illustrating why we need to exclude the
// instantiated type from nest when evaluating the validity of
// a type parameter. Given the declarations
//
//   var _ A[A[string]]
//
//   type A[P any] struct { _ B[P] }
//   type B[P any] struct { _ P }
//
// we want to determine if the type A[A[string]] is valid.
// We start evaluating A[A[string]] outside any type nest:
//
//   A[A[string]]
//         nest =
//         path =
//
// The RHS of A is now evaluated in the A[A[string]] nest:
//
//   struct{_ B[P₁]}
//         nest = A[A[string]]
//         path = A[A[string]]
//
// The struct has a single field of type B[P₁] with which
// we continue:
//
//   B[P₁]
//         nest = A[A[string]]
//         path = A[A[string]]
//
//   struct{_ P₂}
//         nest = A[A[string]]->B[P]
//         path = A[A[string]]->B[P]
//
// Eventutally we reach the type parameter P of type B (P₂):
//
//   P₂
//         nest = A[A[string]]->B[P]
//         path = A[A[string]]->B[P]
//
// The type argument for P of B is the type parameter P of A (P₁).
// It must be evaluated in the type nest that existed when B was
// instantiated:
//
//   P₁
//         nest = A[A[string]]        <== type nest at B's instantiation time
//         path = A[A[string]]->B[P]
//
// If we'd use the current nest it would correspond to the path
// which will be wrong as we will see shortly. P's type argument
// is A[string], which again must be evaluated in the type nest
// that existed when A was instantiated with A[string]. That type
// nest is empty:
//
//   A[string]
//         nest =                     <== type nest at A's instantiation time
//         path = A[A[string]]->B[P]
//
// Evaluation then proceeds as before for A[string]:
//
//   struct{_ B[P₁]}
//         nest = A[string]
//         path = A[A[string]]->B[P]->A[string]
//
// Now we reach B[P] again. If we had not adjusted nest, it would
// correspond to path, and we would find B[P] in nest, indicating
// a cycle, which would clearly be wrong since there's no cycle in
// A[string]:
//
//   B[P₁]
//         nest = A[string]
//         path = A[A[string]]->B[P]->A[string]  <== path contains B[P]!
//
// But because we use the correct type nest, evaluation proceeds without
// errors and we get the evaluation sequence:
//
//   struct{_ P₂}
//         nest = A[string]->B[P]
//         path = A[A[string]]->B[P]->A[string]->B[P]
//   P₂
//         nest = A[string]->B[P]
//         path = A[A[string]]->B[P]->A[string]->B[P]
//   P₁
//         nest = A[string]
//         path = A[A[string]]->B[P]->A[string]->B[P]
//   string
//         nest =
//         path = A[A[string]]->B[P]->A[string]->B[P]
//
// At this point we're done and A[A[string]] and is valid.
