// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "iter"

// If typ is a type parameter, underIs returns the result of typ.underIs(f).
// Otherwise, underIs returns the result of f(typ.Underlying()).
func underIs(typ Type, f func(Type) bool) bool {
	return all(typ, func(_, u Type) bool {
		return f(u)
	})
}

// all reports whether f(t, u) is true for all (type/underlying type)
// pairs in the typeset of t. See [typeset] for details of sequence.
func all(t Type, f func(t, u Type) bool) bool {
	if p, _ := Unalias(t).(*TypeParam); p != nil {
		return p.typeset(f)
	}
	return f(t, t.Underlying())
}

// typeset is an iterator over the (type/underlying type) pairs of the
// specific type terms of the type set implied by t.
// If t is a type parameter, the implied type set is the type set of t's constraint.
// In that case, if there are no specific terms, typeset calls yield with (nil, nil).
// If t is not a type parameter, the implied type set consists of just t.
// In any case, typeset is guaranteed to call yield at least once.
func typeset(t Type) iter.Seq2[Type, Type] {
	return func(yield func(t, u Type) bool) {
		_ = all(t, yield)
	}
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
	for t, u := range typeset(t) {
		if cond != nil {
			if err := cond(t, u); err != nil {
				return nil, err
			}
		}

		if u == nil {
			return nil, typeErrorf("no specific type")
		}

		// If this is the first type we're seeing, we're done.
		if cu == nil {
			ct, cu = t, u
			continue
		}

		// If we've seen a channel before, and we have a channel now, they must be compatible.
		if chu, _ := cu.(*Chan); chu != nil {
			if ch, _ := u.(*Chan); ch != nil {
				if !Identical(chu.elem, ch.elem) {
					return nil, typeErrorf("channels %s and %s have different element types", ct, t)
				}
				// If we have different channel directions, keep the restricted one
				// and complain if they conflict.
				switch {
				case chu.dir == ch.dir:
					// nothing to do
				case chu.dir == SendRecv:
					ct, cu = t, u // switch to restricted channel
				case ch.dir != SendRecv:
					return nil, typeErrorf("channels %s and %s have conflicting directions", ct, t)
				}
				continue
			}
		}

		// Otherwise, the current type must have the same underlying type as all previous types.
		if !Identical(cu, u) {
			return nil, typeErrorf("%s and %s have different underlying types", ct, t)
		}
	}
	return cu, nil
}
