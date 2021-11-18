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
	switch t := t.(type) {
	case *Named:
		return t.under()
	case *TypeParam:
		if tparamIsIface {
			return t.iface()
		}
	}
	return t
}

// If x and y are identical, match returns x.
// If x and y are identical channels but for their direction
// and one of them is unrestricted, match returns the channel
// with the restricted direction.
// In all other cases, match returns nil.
func match(x, y Type) Type {
	// Common case: we don't have channels.
	if Identical(x, y) {
		return x
	}

	// We may have channels that differ in direction only.
	if x, _ := x.(*Chan); x != nil {
		if y, _ := y.(*Chan); y != nil && Identical(x.elem, y.elem) {
			// We have channels that differ in direction only.
			// If there's an unrestricted channel, select the restricted one.
			switch {
			case x.dir == SendRecv:
				return y
			case y.dir == SendRecv:
				return x
			}
		}
	}

	// types are different
	return nil
}

// If typ is a type parameter, structuralType returns the single underlying
// type of all types in the corresponding type constraint if it exists, or
// nil otherwise. If the type set contains only unrestricted and restricted
// channel types (with identical element types), the single underlying type
// is the restricted channel type if the restrictions are always the same.
// If typ is not a type parameter, structuralType returns the underlying type.
func structuralType(typ Type) Type {
	var su Type
	if underIs(typ, func(u Type) bool {
		if u == nil {
			return false
		}
		if su != nil {
			u = match(su, u)
			if u == nil {
				return false
			}
		}
		// su == nil || match(su, u) != nil
		su = u
		return true
	}) {
		return su
	}
	return nil
}
