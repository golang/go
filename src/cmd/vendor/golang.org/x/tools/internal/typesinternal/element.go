// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"fmt"
	"go/types"

	"golang.org/x/tools/go/types/typeutil"
)

// ForEachElement calls f for type T and each type reachable from its
// type through reflection. It does this by recursively stripping off
// type constructors; in addition, for each named type N, the type *N
// is added to the result as it may have additional methods.
//
// The caller must provide an initially empty set used to de-duplicate
// identical types, potentially across multiple calls to ForEachElement.
// (Its final value holds all the elements seen, matching the arguments
// passed to f.)
//
// TODO(adonovan): share/harmonize with go/callgraph/rta.
func ForEachElement(rtypes *typeutil.Map, msets *typeutil.MethodSetCache, T types.Type, f func(types.Type)) {
	var visit func(T types.Type, skip bool)
	visit = func(T types.Type, skip bool) {
		if !skip {
			if seen, _ := rtypes.Set(T, true).(bool); seen {
				return // de-dup
			}

			f(T) // notify caller of new element type
		}

		// Recursion over signatures of each method.
		tmset := msets.MethodSet(T)
		for method := range tmset.Methods() {
			sig := method.Type().(*types.Signature)
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
			visit(sig.Params(), true)  // skip the Tuple
			visit(sig.Results(), true) // skip the Tuple
		}

		switch T := T.(type) {
		case *types.Alias:
			visit(types.Unalias(T), skip) // emulates the pre-Alias behavior

		case *types.Basic:
			// nop

		case *types.Interface:
			// nop---handled by recursion over method set.

		case *types.Pointer:
			visit(T.Elem(), false)

		case *types.Slice:
			visit(T.Elem(), false)

		case *types.Chan:
			visit(T.Elem(), false)

		case *types.Map:
			visit(T.Key(), false)
			visit(T.Elem(), false)

		case *types.Signature:
			if T.Recv() != nil {
				panic(fmt.Sprintf("Signature %s has Recv %s", T, T.Recv()))
			}
			visit(T.Params(), true)  // skip the Tuple
			visit(T.Results(), true) // skip the Tuple

		case *types.Named:
			// A pointer-to-named type can be derived from a named
			// type via reflection.  It may have methods too.
			visit(types.NewPointer(T), false)

			// Consider 'type T struct{S}' where S has methods.
			// Reflection provides no way to get from T to struct{S},
			// only to S, so the method set of struct{S} is unwanted,
			// so set 'skip' flag during recursion.
			visit(T.Underlying(), true) // skip the unnamed type

		case *types.Array:
			visit(T.Elem(), false)

		case *types.Struct:
			for i, n := 0, T.NumFields(); i < n; i++ {
				// TODO(adonovan): document whether or not
				// it is safe to skip non-exported fields.
				visit(T.Field(i).Type(), false)
			}

		case *types.Tuple:
			for i, n := 0, T.Len(); i < n; i++ {
				visit(T.At(i).Type(), false)
			}

		case *types.TypeParam, *types.Union:
			// forEachReachable must not be called on parameterized types.
			panic(T)

		default:
			panic(T)
		}
	}
	visit(T, false)
}
