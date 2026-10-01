// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"go/types"
)

// RecvBase returns the receiver base type (if any) of the method fn,
// and reports whether the receiver is a pointer to it.
//
// The spec says of a method declaration: the receiver's "type must be
// a defined type T or a pointer to a defined type T, possibly followed
// by a list of type parameter names [P1, P2, …] enclosed in square
// brackets. T is called the receiver base type."
//
// The named result is nil if fn is not a method, or if its receiver
// is an anonymous interface or struct type, as in interface{ f() }.
//
// TODO(adonovan): go.dev/issue/81715 proposes this as a method of *types.Func.
func RecvBase(fn *types.Func) (isPtr bool, named *types.Named) {
	// The boolean result appears first to
	// avoid misinterpretation as an 'ok'
	// bool, and because that's the natural
	// reading order in "*Named".

	if recv := fn.Signature().Recv(); recv != nil {
		t := recv.Type()
		if ptr, ok := types.Unalias(t).(*types.Pointer); ok {
			isPtr = true
			t = ptr.Elem()
		}
		named, _ = types.Unalias(t).(*types.Named)
	}
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
