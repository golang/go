// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test common subexpression elimination of loads around other operations.

package codegen

func loadsAroundMemEqual(p *int, s1, s2 string) (int, bool) {
	x := *p
	eq := s1 == s2
	y := *p
	// arm64:"MOVD ZR, R0"
	return x - y, eq
}
