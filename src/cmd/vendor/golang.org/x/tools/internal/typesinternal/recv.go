// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"go/types"
)

// ReceiverNamed returns the named type (if any) associated with the
// type of recv, which may be of the form N or *N, or aliases thereof.
// It also reports whether a Pointer was present.
//
// The named result may be nil if recv is from a method on an
// anonymous interface or struct types or in ill-typed code.
func ReceiverNamed(recv *types.Var) (isPtr bool, named *types.Named) {
	t := recv.Type()
	if ptr, ok := types.Unalias(t).(*types.Pointer); ok {
		isPtr = true
		t = ptr.Elem()
	}
	named, _ = types.Unalias(t).(*types.Named)
	return
}

// Unpointer returns T given *T or an alias thereof.
// For all other types it is the identity function.
// It does not look at underlying types.
// The result may be an alias.
//
// Use this function to strip off the optional pointer on a receiver
// in a field or method selection, without losing the named type
// (which is needed to compute the method set).
//
// See also [typeparams.MustDeref], which removes one level of
// indirection from the type, regardless of named types (analogous to
// a LOAD instruction).
func Unpointer(t types.Type) types.Type {
	if ptr, ok := types.Unalias(t).(*types.Pointer); ok {
		return ptr.Elem()
	}
	return t
}
