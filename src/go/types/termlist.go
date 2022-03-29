// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "bytes"

// A termlist represents the type set represented by the union
// t1 ∪ y2 ∪ ... tn of the type sets of the terms t1 to tn.
// A termlist is in normal form if all terms are disjoint.
// termlist operations don't require the operands to be in
// normal form.
type termlist []*term

// allTermlist represents the set of all types.
// It is in normal form.
var allTermlist = termlist{new(term)}

// String prints the termlist exactly (without normalization).
func (xl termlist) String() string {
	if len(xl) == 0 {
		return "∅"
	}
	var buf bytes.Buffer
	for i, x := range xl {
		if i > 0 {
			buf.WriteString(" ∪ ")
		}
		buf.WriteString(x.String())
	}
	return buf.String()
}

// isEmpty reports whether the termlist xl represents the empty set of types.
func (xl termlist) isEmpty() bool {
	// If there's a non-nil term, the entire list is not empty.
	// If the termlist is in normal form, this requires at most
	// one iteration.
	for _, x := range xl {
		if x != nil {
			return false
		}
	}
	return true
}

// isAll reports whether the termlist xl represents the set of all types.
func (xl termlist) isAll() bool {
	// If there's a 𝓤 term, the entire list is 𝓤.
	// If the termlist is in normal form, this requires at most
	// one iteration.
	for _, x := range xl {
		if x != nil && x.typ == nil {
			return true
		}
	}
	return false
}

// norm returns the normal form of xl.
func (xl termlist) norm() termlist {
	// Quadratic algorithm, but good enough for now.
	// TODO(gri) fix asymptotic performance
	used := make([]bool, len(xl))
	var rl termlist
	for i, xi := range xl {
		if xi == nil || used[i] {
			continue
		}
		for j := i + 1; j < len(xl); j++ {
			xj := xl[j]
			if xj == nil || used[j] {
				continue
			}
			if u1, u2 := xi.union(xj); u2 == nil {
				// If we encounter a 𝓤 term, the entire list is 𝓤.
				// Exit early.
				// (Note that this is not just an optimization;
				// if we continue, we may end up with a 𝓤 term
				// and other terms and the result would not be
				// in normal form.)
				if u1.typ == nil {
					return allTermlist
				}
				xi = u1
				used[j] = true // xj is now unioned into xi - ignore it in future iterations
			}
		}
		rl = append(rl, xi)
	}
	return rl
}

// union returns the union xl ∪ yl.
func (xl termlist) union(yl termlist) termlist {
	return append(xl, yl...).norm()
}

// intersect returns the intersection xl ∩ yl.
func (xl termlist) intersect(yl termlist) termlist {
	if xl.isEmpty() || yl.isEmpty() {
		return nil
	}

	// Quadratic algorithm, but good enough for now.
	// TODO(gri) fix asymptotic performance
	var rl termlist
	for _, x := range xl {
		for _, y := range yl {
			if r := x.intersect(y); r != nil {
				rl = append(rl, r)
			}
		}
	}
	return rl.norm()
}

// equal reports whether xl and yl represent the same type set.
func (xl termlist) equal(yl termlist) bool {
	// TODO(gri) this should be more efficient
	return xl.subsetOf(yl) && yl.subsetOf(xl)
}

// includes reports whether t ∈ xl.
func (xl termlist) includes(t Type) bool {
	for _, x := range xl {
		if x.includes(t) {
			return true
		}
	}
	return false
}

// supersetOf reports whether y ⊆ xl.
func (xl termlist) supersetOf(y *term) bool {
	for _, x := range xl {
		if y.subsetOf(x) {
			return true
		}
	}
	return false
}

// subsetOf reports whether xl ⊆ yl.
func (xl termlist) subsetOf(yl termlist) bool {
	if yl.isEmpty() {
		return xl.isEmpty()
	}

	// each term x of xl must be a subset of yl
	for _, x := range xl {
		if !yl.supersetOf(x) {
			return false // x is not a subset yl
		}
	}
	return true
}
