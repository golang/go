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

// A typeError describes a type error.
type typeError struct {
	format_ string
	args    []any
}

var emptyTypeError typeError

func typeErrorf(format string, args ...any) *typeError {
	if format == "" {
		return &emptyTypeError
	}
	return &typeError{format, args}
}

// format formats a type error as a string.
// check may be nil.
func (err *typeError) format(check *Checker) string {
	return check.sprintf(err.format_, err.args...)
}

// If t is a type parameter, cond is nil, and t's type set contains no channel types,
// commonUnder returns the common underlying type of all types in t's type set if
// it exists, or nil and a type error otherwise.
//
// If t is a type parameter, cond is nil, and there are channel types, t's type set
// must only contain channel types, they must all have the same element types,
// channel directions must not conflict, and commonUnder returns one of the most
// restricted channels. Otherwise, the function returns nil and a type error.
//
// If cond != nil, each pair (t, u) of type and underlying type in t's type set
// must satisfy the condition expressed by cond. If the result of cond is != nil,
// commonUnder returns nil and the type error reported by cond.
// Note that cond is called before any other conditions are checked; specifically
// cond may be called with (nil, nil) if the type set contains no specific types.
//
// If t is not a type parameter, commonUnder behaves as if t was a type parameter
// with the single type t in its type set.
func commonUnder(t Type, cond func(t, u Type) *typeError) (Type, *typeError) {
	var ct, cu Type // type and respective common underlying type
	var err *typeError

	bad := func(format string, args ...any) bool {
		cu = nil
		err = typeErrorf(format, args...)
		return false
	}

	typeset(t, func(t, u Type) bool {
		if cond != nil {
			if err = cond(t, u); err != nil {
				cu = nil
				return false
			}
		}

		if u == nil {
			return bad("no specific type")
		}

		// If this is the first type we're seeing, we're done.
		if cu == nil {
			ct, cu = t, u
			return true
		}

		// If we've seen a channel before, and we have a channel now, they must be compatible.
		if chu, _ := cu.(*Chan); chu != nil {
			if ch, _ := u.(*Chan); ch != nil {
				if !Identical(chu.elem, ch.elem) {
					return bad("channels %s and %s have different element types", ct, t)
				}
				// If we have different channel directions, keep the restricted one
				// and complain if they conflict.
				if chu.dir == SendRecv {
					ct, cu = t, u // switch to current, possibly restricted channel
				} else if chu.dir != ch.dir {
					return bad("channels %s and %s have conflicting directions", ct, t)

				}
				return true
			}
		}

		// Otherwise, the current type must have the same underlying type as all previous types.
		if !Identical(cu, u) {
			return bad("%s and %s have different underlying types", ct, t)
		}

		return true
	})

	return cu, err
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
