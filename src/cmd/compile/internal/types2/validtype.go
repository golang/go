// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// validType verifies that the given type does not "expand" indefinitely
// producing a cycle in the type graph.
// (Cycles involving only alias types, as in "type A = [10]A", are detected
// earlier, via the directCycles and objDecl cycle detection mechanisms.)
func (check *Checker) validType(typ *Named) {
	check.validType0(nopos, typ, nil, nil)
}

// validType0 checks if the given type is valid. If typ is a type parameter
// defined by the enclosing named type in nest, its value is looked up in the
// enclosing type's type argument list if instantiated (otherwise typ's validity
// depends on that uninstantiated enclosing type). Any other type parameter must
// be from an enclosing function or method and can be ignored.
//
// The nest list describes the stack (the "nest in memory") of types which
// contain (or embed in the case of interfaces) other types. For instance, a
// struct named S which contains a field of named type F contains (the memory
// of) F in S, leading to the nest S->F. If a type appears in its own nest
// (say S->F->S) we have an invalid recursive type. The path list is the full
// path of visited named types; it is only needed for error reporting if a cycle
// is found.
//
// validType0 returns a negative result if the argument type is invalid
// and a cycle error is reported.
// If 0 <= result < len(nest), nest[result] is the outermost type which
// binds (instantiates or declares) a type parameter of the argument type;
// i.e., the validity of the argument type depends on the type arguments
// provided at that level.
// If result == len(nest), the type argument is unconditionally valid.
func (check *Checker) validType0(pos syntax.Pos, typ Type, nest, path []*Named) int {
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

	res := len(nest) // assume typ is valid

	switch t := typ.(type) {
	case nil:
		// We should never see a nil type but be conservative and panic
		// only in debug mode.
		if debug {
			panic("validType0(nil)")
		}

	case *Array:
		// An array type is valid if the element type is valid.
		return check.validType0(pos, t.elem, nest, path)

	case *Struct:
		// A struct type is valid if all its field types are valid.
		for _, f := range t.fields {
			d := check.validType0(pos, f.typ, nest, path)
			if d < 0 {
				return d
			}
			// Different fields may depend on type arguments from different
			// depths in nest. Return the nest index of the outermost enclosing
			// type (i.e., the smallest index) which binds (instantiates or declares)
			// a type parameter used by this struct.
			//
			// For example:
			//
			//   type Mid[Q, R any] struct { q Q; r R }
			//   type Outer[P any] Mid[P, int]
			//
			// When checking Mid's struct in nest Outer->Mid[P, int]:
			// field q resolves Q -> P (d == 0, bound at Outer), while
			// field r resolves R -> int (d == 1, bound at Mid[P, int]).
			// Taking min(0, 1) yields 0 for Mid[P, int] indicating that
			// Mid[P, Q] is only conditionally valid (based on how P is
			// ultimately instantiated).
			// Because Mid[P, int] is only conditionally valid, it is
			// also not added to check.valids.
			res = min(res, d)
		}

	case *Union:
		// A union is valid if all its terms are valid.
		for _, t := range t.terms {
			d := check.validType0(pos, t.typ, nest, path)
			if d < 0 {
				return d
			}
			res = min(res, d) // see *Struct case
		}

	case *Interface:
		// An interface is valid if all its embedded types are valid.
		for _, etyp := range t.embeddeds {
			d := check.validType0(pos, etyp, nest, path)
			if d < 0 {
				return d
			}
			res = min(res, d) // see *Struct case
		}

	case *Named:
		// Exit early if we already know t is valid for any nest.
		// This is purely an optimization but it prevents excessive computation
		// times in pathological cases such as testdata/fixedbugs/issue6977.go.
		if check.valids.lookup(t) != nil {
			break
		}

		// If the current type t is also found in nest, (the memory of) t is
		// embedded in itself, indicating an invalid recursive type.
		for _, e := range nest {
			if Identical(e, t) {
				// We have a cycle. If t != t.Origin() then t is an instance of
				// the generic type t.Origin(). Because t is in the nest, t must
				// occur within the definition (RHS) of the generic type t.Origin(),
				// directly or indirectly, after expansion of the RHS.
				// Therefore t.Origin() must be invalid, no matter how it is
				// instantiated, since the instantiation t of t.Origin() happens
				// inside t.Origin()'s RHS and thus is always the same and always
				// present.
				// Therefore we can mark t.Origin()'s RHS as invalid. If t is not an
				// instance of a generic type, t and t.Origin() are the same; otherwise
				// t (and any other instance of t.Origin()) will become invalid when
				// unpacked.
				// Furthermore, because we check all types in a package for validity
				// before type checking is complete, any exported type that is invalid
				// is loaded with fromRHS == Typ[Invalid], so checking its RHS terminates
				// immediately and cannot reach here in a cycle.
				// Thus, if we reach here with a type t, both t and t.Origin() (if
				// different in the first place) must be from the current package;
				// they cannot have been imported.
				// Therefore it is safe to mutate t.Origin().fromRHS; there is no
				// chance for a race condition (the types of the current package
				// are not yet available to other goroutines).
				assert(t.obj.pkg == check.pkg)
				assert(t.Origin().obj.pkg == check.pkg)
				t.Origin().fromRHS = Typ[Invalid]

				// Find the starting point of the cycle and report it.
				// Because each type in nest must also appear in path (see invariant below),
				// type t must be in path since it was found in nest. But not every type in
				// path is in nest: because nest is a pruned subsequence of path, t appears
				// in path at a later (larger or equal) index than in nest. Search again.
				for start, p := range path {
					if Identical(p, t) {
						// Report the error starting with path[start] (firstInSrc == false)
						// rather than the earliest source position on the cycle: path[start:]
						// may traverse instantiated helper types inside valid generic type
						// declarations that appear earlier in the source. For example:
						//
						//   type A[P any] [1]P
						//   type B        A[B]	 // invalid recursive type
						//
						// B is invalid, but A is valid. The cycle path contains both A and B
						// but the error should be on the instantiation A[B] (or B) and not A
						// (which happens to be first in the source in this example).
						// In other words, the issue is not the declaration of A, but rather
						// the instantiation B[C].
						const firstInSrc = false
						check.cycleError(makeObjList(path[start:]), firstInSrc)
						return -1
					}
				}
				panic("cycle start not found")
			}
		}

		// No cycle was found. Check the RHS of t.
		// Every type added to nest is also added to path; thus every type that is in nest
		// must also be in path (invariant). But not every type in path is in nest, since
		// nest may be pruned (see below, *TypeParam case).
		t.Origin().unpack()
		if d := check.validType0(pos, t.Origin().rhs(), append(nest, t), append(path, t)); d < len(nest) {
			// d < 0             : t is invalid
			// 0 <= d < len(nest): t's validity depends on nest[d], which defines a type parameter used by t
			return d
		}

		// d >= len(nest): t's validity does not depend on any type parameters in nest.
		check.valids.add(t) // t is valid for any nest

	case *TypeParam:
		// A type parameter stands for the type (argument) it was instantiated with.
		// Check the corresponding type argument for validity if we are in an
		// instantiated type.
		if d := len(nest) - 1; d >= 0 {
			inst := nest[d] // the current enclosing named type
			// Find the corresponding type argument targ for the type parameter
			// and proceed with checking that type argument.
			for i, tparam := range inst.TypeParams().list() {
				if t == tparam {
					// The type parameter tparam is found, but inst may be an uninstantiated
					// generic type (in which case we have inst.TypeArgs().Len() == 0).
					// Less likely, there may be an unrelated error and type parameter
					// and type argument lists may not match in length.
					// Cover both cases with the if-condition below.
					if i < inst.TypeArgs().Len() {
						targ := inst.TypeArgs().At(i)
						// targ is the type argument with which type parameter tparam
						// was instantiated. targ must be valid in the enclosing
						// type (where inst was instantiated), hence we must check
						// targ's validity in the type nest excluding the current
						// (instantiated) type (see the example at the end of this file).
						// For error reporting we keep the full path.
						res := check.validType0(pos, targ, nest[:d], path)
						// The check.validType0 call with nest[:d] may have
						// overwritten the entry at the current depth d.
						// Restore the entry (was issue go.dev/issue/66323).
						nest[d] = inst
						return res
					}
					// i >= inst.TypeArgs().Len()
					// There is no type argument for tparam; inst is not (or not fully)
					// instantiated (perhaps due to an unrelated error).
					// Example: type T[P any] struct { a P; b ... } // T is not instantiated, there is no targ for P
					// t (== tparam) is valid and defined at nest[d]; per validType0 contract, the result is d.
					return d
				}
			}
		}

	default:
		// Everything else (*Basic, *Pointer, *Slice, *Map, *Chan, *Signature, *Tuple)
		// is either basic or introduces an indirection and thus cannot contribute to
		// an in-memory size/embedding cycle. Nothing to do.
	}

	return res
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
//   type _ = A[A[string]]
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
// If we used the current nest it would correspond to the path
// which would be wrong as we will see shortly. P₁'s type argument
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
// At this point we're done and A[A[string]] is valid.
