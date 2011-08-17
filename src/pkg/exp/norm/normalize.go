// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package norm contains types and functions for normalizing Unicode strings.
package norm

// A Form denotes a canonical representation of Unicode code points.
// The Unicode-defined normalization and equivalence forms are:
//
//   NFC   Unicode Normalization Form C
//   NFD   Unicode Normalization Form D
//   NFKC  Unicode Normalization Form KC
//   NFKD  Unicode Normalization Form KD
//
// For a Form f, this documentation uses the notation f(x) to mean
// the bytes or string x converted to the given form.
// A position n in x is called a boundary if conversion to the form can
// proceed independently on both sides:
//   f(x) == append(f(x[0:n]), f(x[n:])...)
//
// References: http://unicode.org/reports/tr15/ and
// http://unicode.org/notes/tn5/.
type Form int

const (
	NFC Form = iota
	NFD
	NFKC
	NFKD
)

// Bytes returns f(b). May return b if f(b) = b.
func (f Form) Bytes(b []byte) []byte {
	panic("not implemented")
}

// String returns f(s).
func (f Form) String(s string) string {
	panic("not implemented")
}

// IsNormal returns true if b == f(b).
func (f Form) IsNormal(b []byte) bool {
	panic("not implemented")
}

// IsNormalString returns true if s == f(s).
func (f Form) IsNormalString(s string) bool {
	panic("not implemented")
}

// Append returns f(append(out, b...)).
// The buffer out must be empty or equal to f(out).
func (f Form) Append(out, b []byte) []byte {
	panic("not implemented")
}

// AppendString returns f(append(out, []byte(s))).
// The buffer out must be empty or equal to f(out).
func (f Form) AppendString(out []byte, s string) []byte {
	panic("not implemented")
}

// QuickSpan returns a boundary n such that b[0:n] == f(b[0:n]).
// It is not guaranteed to return the largest such n.
func (f Form) QuickSpan(b []byte) int {
	panic("not implemented")
}

// QuickSpanString returns a boundary n such that b[0:n] == f(s[0:n]).
// It is not guaranteed to return the largest such n.
func (f Form) QuickSpanString(s string) int {
	panic("not implemented")
}

// FirstBoundary returns the position i of the first boundary in b.
// It returns len(b), false if b contains no boundaries.
func (f Form) FirstBoundary(b []byte) (i int, ok bool) {
	panic("not implemented")
}

// FirstBoundaryInString return the position i of the first boundary in s.
// It returns len(s), false if s contains no boundaries.
func (f Form) FirstBoundaryInString(s string) (i int, ok bool) {
	panic("not implemented")
}

// LastBoundaryIn returns the position i of the last boundary in b.
// It returns 0, false if b contains no boundary.
func (f Form) LastBoundary(b []byte) (i int, ok bool) {
	panic("not implemented")
}

// LastBoundaryInString returns the position i of the last boundary in s.
// It returns 0, false if s contains no boundary.
func (f Form) LastBoundaryInString(s string) (i int, ok bool) {
	panic("not implemented")
}
