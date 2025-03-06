// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

// under returns the true expanded underlying type.
// If it doesn't exist, the result is Typ[Invalid].
// under must only be called when a type is known
// to be fully set up.
func under(t Type) Type {
	if t := asNamed(t); t != nil {
		return t.under()
	}
	return t.Underlying()
}

// If typ is a type parameter, underIs returns the result of typ.underIs(f).
// Otherwise, underIs returns the result of f(under(typ)).
func underIs(typ Type, f func(Type) bool) bool {
	var ok bool
	typeset(typ, func(_, u Type) bool {
		ok = f(u)
		return ok
	})
	return ok
}

// typeset is an iterator over the (type/underlying type) pairs of the
// specific type terms of the type set implied by t.
// If t is a type parameter, the implied type set is the type set of t's constraint.
// In that case, if there are no specific terms, typeset calls yield with (nil, nil).
// If t is not a type parameter, the implied type set consists of just t.
// In any case, typeset is guaranteed to call yield at least once.
func typeset(t Type, yield func(t, u Type) bool) {
	if p, _ := Unalias(t).(*TypeParam); p != nil {
		p.typeset(yield)
		return
	}
	yield(t, under(t))
}

// TODO(gri) commonUnder, commonUnderOrChan, and Checker.chanElem (expr.go)
//           have a lot of similarities. Maybe we can find common ground
//           between them and distill a better factorization.

// If t is not a type parameter, commonUnder returns the underlying type.
// If t is a type parameter, commonUnder returns the common underlying
// type of all types in its type set if it exists.
// Otherwise the result is nil, and *cause reports the error if a non-nil
// cause is provided.
// The check parameter is only used if *cause reports an error; it may be nil.
func commonUnder(check *Checker, t Type, cause *string) Type {
	var s, su Type

	bad := func(s string) bool {
		if cause != nil {
			*cause = s
		}
		su = nil
		return false
	}

	typeset(t, func(t, u Type) bool {
		if u == nil {
			return bad("no specific type")
		}
		if su != nil && !Identical(su, u) {
			return bad(check.sprintf("%s and %s have different underlying types", s, t))
		}
		// su == nil || Identical(su, u)
		s, su = t, u
		return true
	})

	return su
}

// If t is not a type parameter, commonUnderOrChan returns the underlying type;
// if that type is a channel type it must permit receive operations.
// If t is a type parameter, commonUnderOrChan returns the common underlying
// type of all types in its type set if it exists, or, if the type set contains
// only channel types permitting receive operations and with identical element
// types, commonUnderOrChan returns one of those channel types.
// Otherwise the result is nil, and *cause reports the error if a non-nil cause
// is provided.
// The check parameter is only used if *cause reports an error; it may be nil.
func commonUnderOrChan(check *Checker, t Type, cause *string) Type {
	var s, su Type
	var sc *Chan

	bad := func(s string) bool {
		if cause != nil {
			*cause = s
		}
		su = nil
		return false
	}

	typeset(t, func(t, u Type) bool {
		if u == nil {
			return bad("no specific type")
		}
		c, _ := u.(*Chan)
		if c != nil && c.dir == SendOnly {
			return bad(check.sprintf("receive from send-only channel %s", t))
		}
		if su == nil {
			s, su = t, u
			sc = c // possibly nil
			return true
		}
		// su != nil
		if sc != nil && c != nil {
			if !Identical(sc.elem, c.elem) {
				return bad(check.sprintf("channels with different element types %s and %s", sc.elem, c.elem))
			}
			return true
		}
		// sc == nil
		if !Identical(su, u) {
			return bad(check.sprintf("%s and %s have different underlying types", s, t))
		}
		return true
	})

	return su
}

// If t is not a type parameter, coreType returns the underlying type.
// If t is a type parameter, coreType returns the single underlying
// type of all types in its type set if it exists, or nil otherwise. If the
// type set contains only unrestricted and restricted channel types (with
// identical element types), the single underlying type is the restricted
// channel type if the restrictions are always the same, or nil otherwise.
func coreType(t Type) Type {
	var su Type
	typeset(t, func(_, u Type) bool {
		if u == nil {
			return false
		}
		if su != nil {
			u = match(su, u)
			if u == nil {
				su = nil
				return false
			}
		}
		// su == nil || match(su, u) != nil
		su = u
		return true
	})
	return su
}

// coreString is like coreType but also considers []byte
// and strings as identical. In this case, if successful and we saw
// a string, the result is of type (possibly untyped) string.
func coreString(t Type) Type {
	// This explicit case is needed because otherwise the
	// result would be string if t is an untyped string.
	if !isTypeParam(t) {
		return under(t) // untyped string remains untyped
	}

	var su Type
	hasString := false
	typeset(t, func(_, u Type) bool {
		if u == nil {
			return false
		}
		if isString(u) {
			u = NewSlice(universeByte)
			hasString = true
		}
		if su != nil {
			u = match(su, u)
			if u == nil {
				su = nil
				hasString = false
				return false
			}
		}
		// su == nil || match(su, u) != nil
		su = u
		return true
	})
	if hasString {
		return Typ[String]
	}
	return su
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
