// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/token"
)

// ----------------------------------------------------------------------------
// API

// A Union represents a union of terms.
// A term is a type with a ~ (tilde) flag.
type Union struct {
	types []Type // types are unique
	tilde []bool // if tilde[i] is set, terms[i] is of the form ~T
}

// NewUnion returns a new Union type with the given terms (types[i], tilde[i]).
// The lengths of both arguments must match. An empty union represents the set
// of no types.
func NewUnion(types []Type, tilde []bool) *Union { return newUnion(types, tilde) }

func (u *Union) IsEmpty() bool           { return len(u.types) == 0 }
func (u *Union) NumTerms() int           { return len(u.types) }
func (u *Union) Term(i int) (Type, bool) { return u.types[i], u.tilde[i] }

func (u *Union) Underlying() Type { return u }
func (u *Union) String() string   { return TypeString(u, nil) }

// ----------------------------------------------------------------------------
// Implementation

var emptyUnion = new(Union)

func newUnion(types []Type, tilde []bool) *Union {
	assert(len(types) == len(tilde))
	if len(types) == 0 {
		return emptyUnion
	}
	t := new(Union)
	t.types = types
	t.tilde = tilde
	return t
}

// is reports whether f returned true for all terms (type, tilde) of u.
func (u *Union) is(f func(Type, bool) bool) bool {
	if u.IsEmpty() {
		return false
	}
	for i, t := range u.types {
		if !f(t, u.tilde[i]) {
			return false
		}
	}
	return true
}

// is reports whether f returned true for the underlying types of all terms of u.
func (u *Union) underIs(f func(Type) bool) bool {
	if u.IsEmpty() {
		return false
	}
	for _, t := range u.types {
		if !f(under(t)) {
			return false
		}
	}
	return true
}

func parseUnion(check *Checker, tlist []ast.Expr) Type {
	var types []Type
	var tilde []bool
	for _, x := range tlist {
		t, d := parseTilde(check, x)
		if len(tlist) == 1 && !d {
			return t // single type
		}
		types = append(types, t)
		tilde = append(tilde, d)
	}

	// Ensure that each type is only present once in the type list.
	// It's ok to do this check at the end because it's not a requirement
	// for correctness of the code.
	// Note: This is a quadratic algorithm, but unions tend to be short.
	check.later(func() {
		for i, t := range types {
			t := expand(t)
			if t == Typ[Invalid] {
				continue
			}

			x := tlist[i]
			pos := x.Pos()
			// We may not know the position of x if it was a typechecker-
			// introduced ~T type of a type list entry T. Use the position
			// of T instead.
			// TODO(rfindley) remove this test once we don't support type lists anymore
			if !pos.IsValid() {
				if op, _ := x.(*ast.UnaryExpr); op != nil {
					pos = op.X.Pos()
				}
			}

			u := under(t)
			if tilde[i] && !Identical(u, t) {
				check.errorf(x, _Todo, "invalid use of ~ (underlying type of %s is %s)", t, u)
				continue // don't report another error for t
			}
			if _, ok := u.(*Interface); ok {
				// A single type with a ~ is a single-term union.
				check.errorf(atPos(pos), _Todo, "cannot use interface %s with ~ or inside a union (implementation restriction)", t)
				continue // don't report another error for t
			}

			// Complain about duplicate entries a|a, but also a|~a, and ~a|~a.
			// TODO(gri) We should also exclude myint|~int since myint is included in ~int.
			if includes(types[:i], t) {
				// TODO(rfindley) this currently doesn't print the ~ if present
				check.softErrorf(atPos(pos), _Todo, "duplicate term %s in union element", t)
			}
		}
	})

	return newUnion(types, tilde)
}

func parseTilde(check *Checker, x ast.Expr) (Type, bool) {
	tilde := false
	if op, _ := x.(*ast.UnaryExpr); op != nil && op.Op == token.TILDE {
		x = op.X
		tilde = true
	}
	return check.anyType(x), tilde
}

// intersect computes the intersection of the types x and y,
// A nil type stands for the set of all types; an empty union
// stands for the set of no types.
func intersect(x, y Type) (r Type) {
	// If one of the types is nil (no restrictions)
	// the result is the other type.
	switch {
	case x == nil:
		return y
	case y == nil:
		return x
	}

	// Compute the terms which are in both x and y.
	// TODO(gri) This is not correct as it may not always compute
	//           the "largest" intersection. For instance, for
	//           x = myInt|~int, y = ~int
	//           we get the result myInt but we should get ~int.
	xu, _ := x.(*Union)
	yu, _ := y.(*Union)
	switch {
	case xu != nil && yu != nil:
		// Quadratic algorithm, but good enough for now.
		// TODO(gri) fix asymptotic performance
		var types []Type
		var tilde []bool
		for j, y := range yu.types {
			yt := yu.tilde[j]
			if r, rt := xu.intersect(y, yt); r != nil {
				// Terms x[i] and y[j] match: Select the one that
				// is not a ~t because that is the intersection
				// type. If both are ~t, they are identical:
				//  T ∩  T =  T
				//  T ∩ ~t =  T
				// ~t ∩  T =  T
				// ~t ∩ ~t = ~t
				types = append(types, r)
				tilde = append(tilde, rt)
			}
		}
		return newUnion(types, tilde)

	case xu != nil:
		if r, _ := xu.intersect(y, false); r != nil {
			return y
		}

	case yu != nil:
		if r, _ := yu.intersect(x, false); r != nil {
			return x
		}

	default: // xu == nil && yu == nil
		if Identical(x, y) {
			return x
		}
	}

	return emptyUnion
}

// includes reports whether typ is in list.
func includes(list []Type, typ Type) bool {
	for _, e := range list {
		if Identical(typ, e) {
			return true
		}
	}
	return false
}

// intersect computes the intersection of the union u and term (y, yt)
// and returns the intersection term, if any. Otherwise the result is
// (nil, false).
func (u *Union) intersect(y Type, yt bool) (Type, bool) {
	under_y := under(y)
	for i, x := range u.types {
		xt := u.tilde[i]
		// determine which types xx, yy to compare
		xx := x
		if yt {
			xx = under(x)
		}
		yy := y
		if xt {
			yy = under_y
		}
		if Identical(xx, yy) {
			//  T ∩  T =  T
			//  T ∩ ~t =  T
			// ~t ∩  T =  T
			// ~t ∩ ~t = ~t
			return xx, xt && yt
		}
	}
	return nil, false
}
