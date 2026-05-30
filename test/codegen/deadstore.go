// asmcheck

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type S struct {
	a, b, c, d, e int
}

func f1(s *S) {
	// amd64:-`MOVUPS X15`
	// arm64:`FSTPQ` -`STP \(ZR, ZR\)` -`MOVD ZR`
	*s = S{}
	// arm64:`MOVD \$7` `MOVD R[0-9]+, 32\(R[0-9]+\)`
	*s = S{a: 3, b: 4, c: 5, d: 6, e: 7}
}

func f2(s *S) {
	// amd64:`MOVQ \$3`
	// arm64:-`FSTPQ`
	*s = S{a: 1, b: 2, c: 3, d: 4, e: 5}
	// amd64:-`MOVQ`
	s.a = 3
	// amd64:`MOVQ \$4`
	s.b = 4
	// amd64:`MOVQ \$5`
	s.c = 5
	// amd64:`MOVQ \$6`
	s.d = 6
	// amd64:`MOVQ \$7`
	s.e = 7
}
