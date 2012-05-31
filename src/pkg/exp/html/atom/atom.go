// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package atom provides integer codes (also known as atoms) for a fixed set of
// frequently occurring HTML strings: lower-case tag names and attribute keys
// such as "p" and "id".
//
// Sharing an atom's string representation between all elements with the same
// tag can result in fewer string allocations when tokenizing and parsing HTML.
// Integer comparisons are also generally faster than string comparisons.
//
// An atom's particular code (such as atom.Div == 63) is not guaranteed to
// stay the same between versions of this package. Neither is any ordering
// guaranteed: whether atom.H1 < atom.H2 may also change. The codes are not
// guaranteed to be dense. The only guarantees are that e.g. looking up "div"
// will yield atom.Div, calling atom.Div.String will return "div", and
// atom.Div != 0.
package atom

// Atom is an integer code for a string. The zero value maps to "".
type Atom int

// String returns the atom's string representation.
func (a Atom) String() string {
	if a <= 0 || a > max {
		return ""
	}
	return table[a]
}

// Lookup returns the atom whose name is s. It returns zero if there is no
// such atom.
func Lookup(s []byte) Atom {
	if len(s) == 0 {
		return 0
	}
	if len(s) == 1 {
		x := s[0]
		if x < 'a' || x > 'z' {
			return 0
		}
		return oneByteAtoms[x-'a']
	}
	// Binary search for the atom. Unlike sort.Search, this returns early on an exact match.
	// TODO: this could be optimized further. For example, lo and hi could be initialized
	// from s[0]. Separately, all the "onxxx" atoms could be moved into their own table.
	lo, hi := Atom(1), 1+max
	for lo < hi {
		mid := (lo + hi) / 2
		if cmp := compare(s, table[mid]); cmp == 0 {
			return mid
		} else if cmp > 0 {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return 0
}

// String returns a string whose contents are equal to s. In that sense, it is
// equivalent to string(s), but may be more efficient.
func String(s []byte) string {
	if a := Lookup(s); a != 0 {
		return a.String()
	}
	return string(s)
}

// compare is like bytes.Compare, except that it takes one []byte argument and
// one string argument, and returns negative/0/positive instead of -1/0/+1.
func compare(s []byte, t string) int {
	n := len(s)
	if n > len(t) {
		n = len(t)
	}
	for i, si := range s[:n] {
		ti := t[i]
		switch {
		case si > ti:
			return +1
		case si < ti:
			return -1
		}
	}
	return len(s) - len(t)
}
