// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Helper functions exported for the compiler.
// Do not use internally.

package types2

// If t is a pointer, AsPointer returns that type, otherwise it returns nil.
func AsPointer(t Type) *Pointer {
	u, _ := t.Underlying().(*Pointer)
	return u
}

// RangeKeyVal returns the key and value types for a range over typ.
// It panics if range over typ is invalid.
func RangeKeyVal(typ Type) (Type, Type) {
	key, val, _, ok := rangeKeyVal(nil, typ, nil)
	assert(ok)
	return key, val
}
