// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"fmt"
	"go/types"
)

// ForEachElement calls f for type T and each type reachable from its
// type through reflection. It does this by recursively stripping off
// type constructors; in addition, for each named type N, the type *N
// is added to the result as it may have additional methods.
//
// The access argument passed to f indicates whether the type is
// inaccessible to reflection (for example, intermediate tuple types
// or underlying types of named types).
//
// The result of f indicates whether the caller has seen this type
// already, so we can prune the traversal.
//
// methodSetOf abstracts (*typeutil.MethodSetCache).MethodSet,
// avoiding an import cycle.
func ForEachElement(methodSetOf func(types.Type) *types.MethodSet, T types.Type, f func(T types.Type, access bool) bool) {
	var visit func(T types.Type, access bool)
	visit = func(T types.Type, access bool) {
		if f(T, access) {
			return // duplicate; prune descent
		}

		// Recursion over signatures of each method.
		tmset := methodSetOf(T)
		for method := range tmset.Methods() {
			sig := method.Type().(*types.Signature)
			if sig.TypeParams() != nil {
				continue // skip type-parameterized methods
			}

			// It is tempting to call visit(sig, false)
			// but, as noted in golang.org/cl/65450043,
			// the Signature.Recv field is ignored by
			// types.Identical and typeutil.Map, which
			// is confusing at best.
			//
			// More importantly, the true signature rtype
			// reachable from a method using reflection
			// has no receiver but an extra ordinary parameter.
			// For the Read method of io.Reader we want:
			//   func(Reader, []byte) (int, error)
			// but here sig is:
			//   func([]byte) (int, error)
			// with .Recv = Reader (though it is hard to
			// notice because it doesn't affect Signature.String
			// or types.Identical).
			//
			// TODO(adonovan): construct and visit the correct
			// non-method signature with an extra parameter
			// (though since unnamed func types have no methods
			// there is essentially no actual demand for this).
			//
			// TODO(adonovan): document whether or not it is
			// safe to skip non-exported methods (as RTA does).
			visit(sig.Params(), false)  // the Tuple is inaccessible
			visit(sig.Results(), false) // the Tuple is inaccessible
		}

		switch T := T.(type) {
		case *types.Alias:
			visit(types.Unalias(T), access) // emulates the pre-Alias behavior

		case *types.Basic:
			// nop

		case *types.Interface:
			// nop---handled by recursion over method set.

		case *types.Pointer:
			visit(T.Elem(), true)

		case *types.Slice:
			visit(T.Elem(), true)

		case *types.Chan:
			visit(T.Elem(), true)

		case *types.Map:
			visit(T.Key(), true)
			visit(T.Elem(), true)

		case *types.Signature:
			if T.Recv() != nil {
				panic(fmt.Sprintf("Signature %s has Recv %s", T, T.Recv()))
			}
			visit(T.Params(), false)  // the Tuple is inaccessible
			visit(T.Results(), false) // the Tuple is inaccessible

		case *types.Named:
			// A pointer-to-named type can be derived from a named
			// type via reflection.  It may have methods too.
			visit(types.NewPointer(T), true)

			// Consider 'type T struct{S}' where S has methods.
			// Reflection provides no way to get from T to struct{S},
			// only to S, so the method set of struct{S} is unwanted,
			// so mark it inaccessible during recursion.
			visit(T.Underlying(), false) // skip the unnamed type

		case *types.Array:
			visit(T.Elem(), true)

		case *types.Struct:
			for i, n := 0, T.NumFields(); i < n; i++ {
				// TODO(adonovan): document whether or not
				// it is safe to skip non-exported fields.
				visit(T.Field(i).Type(), true)
			}

		case *types.Tuple:
			for i, n := 0, T.Len(); i < n; i++ {
				visit(T.At(i).Type(), true)
			}

		case *types.TypeParam, *types.Union:
			// forEachReachable must not be called on parameterized types.
			panic(fmt.Sprintf("ForEachElement called on type containing %T", T))

		default:
			panic(fmt.Sprintf("ForEachElement called on unexpected type %T", T))
		}
	}
	visit(T, true)
}
