// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

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

// Convenience converters

func toBasic(t Type) *Basic {
	u, _ := under(t).(*Basic)
	return u
}

func toArray(t Type) *Array {
	u, _ := under(t).(*Array)
	return u
}

func toSlice(t Type) *Slice {
	u, _ := under(t).(*Slice)
	return u
}

func toStruct(t Type) *Struct {
	u, _ := under(t).(*Struct)
	return u
}

func toPointer(t Type) *Pointer {
	u, _ := under(t).(*Pointer)
	return u
}

func toSignature(t Type) *Signature {
	u, _ := under(t).(*Signature)
	return u
}

func toInterface(t Type) *Interface {
	u, _ := under(t).(*Interface)
	return u
}

// If the argument to asNamed, or asTypeParam is of the respective type
// (possibly after expanding resolving a *Named type), these methods return that type.
// Otherwise the result is nil.

func asNamed(t Type) *Named {
	e, _ := t.(*Named)
	if e != nil {
		e.resolve(nil)
	}
	return e
}

func asTypeParam(t Type) *TypeParam {
	u, _ := under(t).(*TypeParam)
	return u
}

// Exported for the compiler.

func AsPointer(t Type) *Pointer     { return toPointer(t) }
func AsNamed(t Type) *Named         { return asNamed(t) }
func AsSignature(t Type) *Signature { return toSignature(t) }
func AsInterface(t Type) *Interface { return toInterface(t) }
func AsTypeParam(t Type) *TypeParam { return asTypeParam(t) }
