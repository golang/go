// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package atom provides integer codes (also known as atoms) for a fixed set of
// frequently occurring HTML strings: lower-case tag names and attribute keys
// such as "p" and "id".
//
// Sharing an atom's name between all elements with the same tag can result in
// fewer string allocations when tokenizing and parsing HTML. Integer
// comparisons are also generally faster than string comparisons.
//
// The value of an atom's particular code is not guaranteed to stay the same
// between versions of this package. Neither is any ordering guaranteed:
// whether atom.H1 < atom.H2 may also change. The codes are not guaranteed to
// be dense. The only guarantees are that e.g. looking up "div" will yield
// atom.Div, calling atom.Div.String will return "div", and atom.Div != 0.
package atom

// The hash function must be the same as the one used in gen.go
func hash(s []byte) (h uint32) {
	for i := 0; i < len(s); i++ {
		h = h<<5 ^ h>>27 ^ uint32(s[i])
	}
	return h
}

// Atom is an integer code for a string. The zero value maps to "".
type Atom int

// String returns the atom's name.
func (a Atom) String() string {
	if 0 <= a && a < Atom(len(table)) {
		return table[a]
	}
	return ""
}

// Lookup returns the atom whose name is s. It returns zero if there is no
// such atom.
func Lookup(s []byte) Atom {
	if len(s) == 0 || len(s) > maxLen {
		return 0
	}
	if len(s) == 1 {
		x := s[0]
		if x < 'a' || x > 'z' {
			return 0
		}
		return oneByteAtoms[x-'a']
	}
	hs := hash(s)
	// Binary search for hs. Unlike sort.Search, this returns early on an exact match.
	// A loop invariant is that len(table[i]) == len(s) for all i in [lo, hi).
	lo := Atom(loHi[len(s)])
	hi := Atom(loHi[len(s)+1])
	for lo < hi {
		mid := (lo + hi) / 2
		if ht := hashes[mid]; hs == ht {
			// The gen.go program ensures that each atom's name has a distinct hash.
			// However, arbitrary strings may collide with the atom's name. We have
			// to check that string(s) == table[mid].
			t := table[mid]
			for i, si := range s {
				if si != t[i] {
					return 0
				}
			}
			return mid
		} else if hs > ht {
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
