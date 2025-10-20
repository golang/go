// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// validType verifies that the given type does not "expand" indefinitely
// producing a cycle in the type graph.
// (Cycles involving alias types, as in "type A = [10]A" are detected
// earlier, via the objDecl cycle detection mechanism.)
func (check *Checker) validType(typ *Named) {
	check.validType0(nopos, typ, nil, nil)
}

// validType0 checks if the given type is valid. If typ is a type parameter
// its value is looked up in the type argument list of the instantiated
// (enclosing) type, if it exists. Otherwise the type parameter must be from
// an enclosing function and can be ignored.
// The nest list describes the stack (the "nest in memory") of types which
// contain (or embed in the case of interfaces) other types. For instance, a
// struct named S which contains a field of named type F contains (the memory
// of) F in S, leading to the nest S->F. If a type appears in its own nest
// (say S->F->S) we have an invalid recursive type. The path list is the full
// path of named types in a cycle, it is only needed for error reporting.
func (check *Checker) validType0(pos syntax.Pos, typ Type, nest, path []*Named) bool {
	typ = Unalias(typ)

	if check.conf.Trace {
		if t, _ := typ.(*Named); t != nil && t.obj != nil /* obj should always exist but be conservative */ {
			pos = t.obj.pos
		}
		check.indent++
		check.trace(pos, "validType(%s) nest %v, path %v", typ, pathString(makeObjList(nest)), pathString(makeObjList(path)))
		defer func() {
			check.indent--
		}()
	}

	switch t := typ.(type) {
	case nil:
		// We should never see a nil type but be conservative and panic
		// only in debug mode.
		if debug {
			panic("validType0(nil)")
		}

	case *Array:
		return check.validType0(pos, t.elem, nest, path)

	case *Struct:
		for _, f := range t.fields {
			if !check.validType0(pos, f.typ, nest, path) {
				return false
			}
		}

	case *Union:
		for _, t := range t.terms {
			if !check.validType0(pos, t.typ, nest, path) {
				return false
			}
		}

	case *Interface:
		for _, etyp := range t.embeddeds {
			if !check.validType0(pos, etyp, nest, path) {
				return false
			}
		}

	case *Named:
		// TODO(gri) The optimization below is incorrect (see go.dev/issue/65711):
		//           in that issue `type A[P any] [1]P` is a valid type on its own
		//           and the (uninstantiated) A is recorded in check.valids. As a
		//           consequence, when checking the remaining declarations, which
		//           are not valid, the validity check ends prematurely because A
		//           is considered valid, even though its validity depends on the
		//           type argument provided to it.
		//
		//           A correct optimization is important for pathological cases.
		//           Keep code around for reference until we found an optimization.
		//
		// // Exit early if we already know t is valid.
		// // This is purely an optimization but it prevents excessive computation
		// // times in pathological cases such as testdata/fixedbugs/issue6977.go.
		// // (Note: The valids map could also be allocated locally, once for each
		// // validType call.)
		// if check.valids.lookup(t) != nil {
		// 	break
		// }

		// If the current type t is also found in nest, (the memory of) t is
		// embedded in itself, indicating an invalid recursive type.
		for _, e := range nest {
			if Identical(e, t) {
				// We have a cycle. If t != t.Origin() then t is an instance of
				// the generic type t.Origin(). Because t is in the nest, t must
				// occur within the definition (RHS) of the generic type t.Origin(),
				// directly or indirectly, after expansion of the RHS.
				// Therefore t.Origin() must be invalid, no matter how it is
				// instantiated since the instantiation t of t.Origin() happens
				// inside t.Origin()'s RHS and thus is always the same and always
				// present.
				// Therefore we can mark the underlying of both t and t.Origin()
				// as invalid. If t is not an instance of a generic type, t and
				// t.Origin() are the same.
				// Furthermore, because we check all types in a package for validity
				// before type checking is complete, any exported type that is invalid
				// will have an invalid underlying type and we can't reach here with
				// such a type (invalid types are excluded above).
				// Thus, if we reach here with a type t, both t and t.Origin() (if
				// different in the first place) must be from the current package;
				// they cannot have been imported.
				// Therefore it is safe to change their underlying types; there is
				// no chance for a race condition (the types of the current package
				// are not yet available to other goroutines).
				assert(t.obj.pkg == check.pkg)
				assert(t.Origin().obj.pkg == check.pkg)

				// let t become invalid when it is unpacked
				t.Origin().fromRHS = Typ[Invalid]

				// Find the starting point of the cycle and report it.
				// Because each type in nest must also appear in path (see invariant below),
				// type t must be in path since it was found in nest. But not every type in path
				// is in nest. Specifically t may appear in path with an earlier index than the
				// index of t in nest. Search again.
				for start, p := range path {
					if Identical(p, t) {
						check.cycleError(makeObjList(path[start:]), 0)
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
		if !check.validType0(pos, t.Origin().fromRHS, append(nest, t), append(path, t)) {
			return false
		}

		// see TODO above
		// check.valids.add(t) // t is valid

	case *TypeParam:
		// A type parameter stands for the type (argument) it was instantiated with.
		// Check the corresponding type argument for validity if we are in an
		// instantiated type.
		if d := len(nest) - 1; d >= 0 {
			inst := nest[d] // the type instance
			// Find the corresponding type argument for the type parameter
			// and proceed with checking that type argument.
			for i, tparam := range inst.TypeParams().list() {
				// The type parameter and type argument lists should
				// match in length but be careful in case of errors.
				if t == tparam && i < inst.TypeArgs().Len() {
					targ := inst.TypeArgs().At(i)
					// The type argument must be valid in the enclosing
					// type (where inst was instantiated), hence we must
					// check targ's validity in the type nest excluding
					// the current (instantiated) type (see the example
					// at the end of this file).
					// For error reporting we keep the full path.
					res := check.validType0(pos, targ, nest[:d], path)
					// The check.validType0 call with nest[:d] may have
					// overwritten the entry at the current depth d.
					// Restore the entry (was issue go.dev/issue/66323).
					nest[d] = inst
					return res
				}
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
// Eventually we reach the type parameter P of type B (P₂):
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
