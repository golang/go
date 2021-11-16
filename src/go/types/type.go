// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// A Type represents a type of Go.
// All types implement the Type interface.
type Type interface {
	// Underlying returns the underlying type of a type
	// w/o following forwarding chains. Only used by
	// client packages.
	Underlying() Type

	// String returns a string representation of a type.
	String() string
}

// under returns the true expanded underlying type.
// If it doesn't exist, the result is Typ[Invalid].
// under must only be called when a type is known
// to be fully set up.
func under(t Type) Type {
	if n := asNamed(t); n != nil {
		return n.under()
	}
	return t
}

// If typ is a type parameter, structuralType returns the single underlying
// type of all types in the corresponding type constraint if it exists,
// or nil otherwise. If typ is not a type parameter, structuralType returns
// the underlying type.
func structuralType(typ Type) Type {
	var su Type
	if underIs(typ, func(u Type) bool {
		if su != nil && !Identical(su, u) {
			return false
		}
		// su == nil || Identical(su, u)
		su = u
		return true
	}) {
		return su
	}
	return nil
}

// structuralString is like structuralType but also considers []byte
// and string as "identical". In this case, if successful, the result
// is always []byte.
func structuralString(typ Type) Type {
	var su Type
	if underIs(typ, func(u Type) bool {
		if isString(u) {
			u = NewSlice(universeByte)
		}
		if su != nil && !Identical(su, u) {
			return false
		}
		// su == nil || Identical(su, u)
		su = u
		return true
	}) {
		return su
	}
	return nil
}

// If t is a defined type, asNamed returns that type (possibly after resolving it), otherwise it returns nil.
func asNamed(t Type) *Named {
	e, _ := t.(*Named)
	if e != nil {
		e.resolve(nil)
	}
	return e
}

// If t is a type parameter, asTypeParam returns that type, otherwise it returns nil.
func asTypeParam(t Type) *TypeParam {
	u, _ := under(t).(*TypeParam)
	return u
}
