// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// ----------------------------------------------------------------------------
// API

// A Union represents a union of terms.
// A term is a type, possibly with a ~ (tilde) indication.
type Union struct {
	terms []Type // terms are unique
	tilde []bool // if tilde[i] is set, terms[i] is of the form ~T
}

func NewUnion(terms []Type, tilde []bool) Type { return newUnion(terms, tilde) }

func (u *Union) NumTerms() int           { return len(u.terms) }
func (u *Union) Term(i int) (Type, bool) { return u.terms[i], u.tilde[i] }

func (u *Union) Underlying() Type { return u }
func (u *Union) String() string   { return TypeString(u, nil) }

// ----------------------------------------------------------------------------
// Implementation

func newUnion(terms []Type, tilde []bool) Type {
	assert(len(terms) == len(tilde))
	if terms == nil {
		return nil
	}
	t := new(Union)
	t.terms = terms
	t.tilde = tilde
	return t
}

func parseUnion(check *Checker, tlist []syntax.Expr) Type {
	var terms []Type
	var tilde []bool
	for _, x := range tlist {
		t, d := parseTilde(check, x)
		if len(tlist) == 1 && !d {
			return t // single type
		}
		terms = append(terms, t)
		tilde = append(tilde, d)
	}

	// Ensure that each type is only present once in the type list.
	// It's ok to do this check at the end because it's not a requirement
	// for correctness of the code.
	// Note: This is a quadratic algorithm, but unions tend to be short.
	check.later(func() {
		for i, t := range terms {
			t := expand(t)
			if t == Typ[Invalid] {
				continue
			}

			x := tlist[i]
			pos := syntax.StartPos(x)
			// We may not know the position of x if it was a typechecker-
			// introduced ~T type of a type list entry T. Use the position
			// of T instead.
			// TODO(gri) remove this test once we don't support type lists anymore
			if !pos.IsKnown() {
				if op, _ := x.(*syntax.Operation); op != nil {
					pos = syntax.StartPos(op.X)
				}
			}

			u := under(t)
			if tilde[i] {
				// TODO(gri) enable this check once we have converted tests
				// if !Identical(u, t) {
				// 	check.errorf(x, "invalid use of ~ (underlying type of %s is %s)", t, u)
				// }
			}
			if _, ok := u.(*Interface); ok {
				check.errorf(pos, "cannot use interface %s with ~ or inside a union (implementation restriction)", t)
			}

			// Complain about duplicate entries a|a, but also a|~a, and ~a|~a.
			if includes(terms[:i], t) {
				// TODO(gri) this currently doesn't print the ~ if present
				check.softErrorf(pos, "duplicate term %s in union element", t)
			}
		}
	})

	return newUnion(terms, tilde)
}

func parseTilde(check *Checker, x syntax.Expr) (Type, bool) {
	tilde := false
	if op, _ := x.(*syntax.Operation); op != nil && op.Op == syntax.Tilde {
		x = op.X
		tilde = true
	}
	return check.anyType(x), tilde
}
