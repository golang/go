// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeutil

// This file defines utilities for user interfaces that display types.

import "go/types"

// IntuitiveMethodSet returns the intuitive method set of a type T,
// which is the set of methods you can call on an addressable value of
// that type.
//
// The result always contains MethodSet(T), and is exactly MethodSet(T)
// for interface types and for pointer-to-concrete types.
// For all other concrete types T, the result additionally
// contains each method belonging to *T if there is no identically
// named method on T itself.
//
// This corresponds to user intuition about method sets;
// this function is intended only for user interfaces.
//
// The order of the result is as for types.MethodSet(T).
func IntuitiveMethodSet(T types.Type, msets *MethodSetCache) []*types.Selection {
	isPointerToConcrete := func(T types.Type) bool {
		ptr, ok := T.(*types.Pointer)
		return ok && !types.IsInterface(ptr.Elem())
	}

	var result []*types.Selection
	mset := msets.MethodSet(T)
	if types.IsInterface(T) || isPointerToConcrete(T) {
		for i, n := 0, mset.Len(); i < n; i++ {
			result = append(result, mset.At(i))
		}
	} else {
		// T is some other concrete type.
		// Report methods of T and *T, preferring those of T.
		pmset := msets.MethodSet(types.NewPointer(T))
		for i, n := 0, pmset.Len(); i < n; i++ {
			meth := pmset.At(i)
			if m := mset.Lookup(meth.Obj().Pkg(), meth.Obj().Name()); m != nil {
				meth = m
			}
			result = append(result, meth)
		}

	}
	return result
}
