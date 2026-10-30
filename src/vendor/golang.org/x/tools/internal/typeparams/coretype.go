// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"fmt"
	"go/types"

	"golang.org/x/tools/internal/moreiters"
)

// CoreType returns the core type of T or nil if T does not have a core type.
//
// As of Go 1.25, the notion of a core type has been removed from the
// language spec; see https://go.dev/blog/coretypes for details.
//
// The spec now expresses the semantics of some operations (with
// perhaps more in future) terms of computations over type sets, which
// may permit them to use operands that do not have a CoreType; use
// [TypeSetOf] in such cases.
//
// However, the spec continues to define other operations in a way
// that is equivalent to the old core-type concept. This function is
// provided for such cases.
func CoreType(T types.Type) types.Type {
	U := T.Underlying()
	if _, ok := U.(*types.Interface); !ok {
		return U // for non-interface types,
	}

	ts := TypeSetOf(U)
	if ts.Len() == 0 {
		// Unrestricted (no specific terms), empty, or invalid type set.
		return nil // no core type.
	}
	seq := ts.UnderlyingTypes()

	// Check whether all underlying types are identical.
	// Channels may be identical modulo direction, so
	// if not, we check them as a second pass.

	U = ts.Term(0).Type().Underlying()
	if moreiters.Every(seq, func(u types.Type) bool { return types.Identical(U, u) }) {
		// From the deprecated core types spec:
		// "There is a single type U which is the underlying type of all types in the type set of T"
		return U
	}

	// From the deprecated core types spec:
	// "the type chan E if T contains only bidirectional channels, or the type chan<- E or
	// <-chan E depending on the direction of the directional channels present."
	//
	// (This pass revisits the terms identical to the first one;
	// that is harmless, and in practice there is usually only one.)
	var ch *types.Chan
	for u := range seq {
		curr, ok := u.(*types.Chan)
		if !ok {
			return nil // no core type as terms are not identical and not all channels.
		}
		if ch == nil {
			ch = curr
			continue
		}
		if !types.Identical(ch.Elem(), curr.Elem()) {
			return nil // channel elements are not identical.
		}
		if ch.Dir() == types.SendRecv {
			// ch is bidirectional. We can safely always use curr's direction.
			ch = curr
		} else if curr.Dir() != types.SendRecv && ch.Dir() != curr.Dir() {
			// ch and curr are not bidirectional and not the same direction.
			return nil
		}
	}
	return ch
}

// Deref returns the type of the variable pointed to by t,
// if t's core type is a pointer; otherwise it returns t.
//
// Do not assume that Deref(T)==T implies T is not a pointer:
// consider "type T *T", for example.
//
// TODO(adonovan): ideally this would live in typesinternal, but that
// creates an import cycle. Move there when we melt this package down.
func Deref(t types.Type) types.Type {
	if ptr, ok := CoreType(t).(*types.Pointer); ok {
		return ptr.Elem()
	}
	return t
}

// MustDeref returns the type of the variable pointed to by t.
// It panics if t's core type is not a pointer.
//
// TODO(adonovan): ideally this would live in typesinternal, but that
// creates an import cycle. Move there when we melt this package down.
func MustDeref(t types.Type) types.Type {
	if ptr, ok := CoreType(t).(*types.Pointer); ok {
		return ptr.Elem()
	}
	panic(fmt.Sprintf("%v is not a pointer", t))
}
