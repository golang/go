// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type S struct {
	a, b, c, d, e int
}

func f1(s *S) {
	// amd64:-`MOVUPS`
	// arm64:-`STP` -`MOVD`
	*s = S{}
	*s = S{a: 3, b: 4, c: 5, d: 6, e: 7}
}

func f2(s *S) {
	// amd64:-`MOVUPS`
	// arm64:-`MOVD` -`FSTPQ`
	*s = S{a: 1, b: 2, c: 3, d: 4, e: 5}
	s.a = 3
	s.b = 4
	s.c = 5
	s.d = 6
	s.e = 7
}
