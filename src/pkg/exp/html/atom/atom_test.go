// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atom

import (
	"sort"
	"testing"
)

func TestKnown(t *testing.T) {
	for _, s := range testAtomList {
		if atom := Lookup([]byte(s)); atom.String() != s {
			t.Errorf("Lookup(%q) = %#x (%q)", s, uint32(atom), atom.String())
		}
	}
}

func TestHits(t *testing.T) {
	for _, a := range table {
		if a == 0 {
			continue
		}
		got := Lookup([]byte(a.String()))
		if got != a {
			t.Errorf("Lookup(%q) = %#x, want %#x", a.String(), uint32(got), uint32(a))
		}
	}
}

func TestMisses(t *testing.T) {
	testCases := []string{
		"",
		"\x00",
		"\xff",
		"A",
		"DIV",
		"Div",
		"dIV",
		"aa",
		"a\x00",
		"ab",
		"abb",
		"abbr0",
		"abbr ",
		" abbr",
		" a",
		"acceptcharset",
		"acceptCharset",
		"accept_charset",
		"h0",
		"h1h2",
		"h7",
		"onClick",
		"λ",
		// The following string has the same hash (0xa1d7fab7) as "onmouseover".
		"\x00\x00\x00\x00\x00\x50\x18\xae\x38\xd0\xb7",
	}
	for _, tc := range testCases {
		got := Lookup([]byte(tc))
		if got != 0 {
			t.Errorf("Lookup(%q): got %d, want 0", tc, got)
		}
	}
}

func TestForeignObject(t *testing.T) {
	const (
		afo = Foreignobject
		afO = ForeignObject
		sfo = "foreignobject"
		sfO = "foreignObject"
	)
	if got := Lookup([]byte(sfo)); got != afo {
		t.Errorf("Lookup(%q): got %#v, want %#v", sfo, got, afo)
	}
	if got := Lookup([]byte(sfO)); got != afO {
		t.Errorf("Lookup(%q): got %#v, want %#v", sfO, got, afO)
	}
	if got := afo.String(); got != sfo {
		t.Errorf("Atom(%#v).String(): got %q, want %q", afo, got, sfo)
	}
	if got := afO.String(); got != sfO {
		t.Errorf("Atom(%#v).String(): got %q, want %q", afO, got, sfO)
	}
}

func BenchmarkLookup(b *testing.B) {
	sortedTable := make([]string, 0, len(table))
	for _, a := range table {
		if a != 0 {
			sortedTable = append(sortedTable, a.String())
		}
	}
	sort.Strings(sortedTable)

	x := make([][]byte, 1000)
	for i := range x {
		x[i] = []byte(sortedTable[i%len(sortedTable)])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range x {
			Lookup(s)
		}
	}
}
