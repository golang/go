// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.5

package typeutil

// This file defines utilities for user interfaces that display types.

import "golang.org/x/tools/go/types"

// IntuitiveMethodSet returns the intuitive method set of a type, T.
//
// The result contains MethodSet(T) and additionally, if T is a
// concrete type, methods belonging to *T if there is no identically
// named method on T itself.  This corresponds to user intuition about
// method sets; this function is intended only for user interfaces.
//
// The order of the result is as for types.MethodSet(T).
//
func IntuitiveMethodSet(T types.Type, msets *MethodSetCache) []*types.Selection {
	var result []*types.Selection
	mset := msets.MethodSet(T)
	if _, ok := T.Underlying().(*types.Interface); ok {
		for i, n := 0, mset.Len(); i < n; i++ {
			result = append(result, mset.At(i))
		}
	} else {
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
