// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests phi implementation

package main

import "testing"

func phiOverwrite_ssa() int {
	var n int
	for i := 0; i < 10; i++ {
		if i == 6 {
			break
		}
		n = i
	}
	return n
}

func phiOverwrite(t *testing.T) {
	want := 5
	got := phiOverwrite_ssa()
	if got != want {
		t.Errorf("phiOverwrite_ssa()= %d, got %d", want, got)
	}
}

func phiOverwriteBig_ssa() int {
	var a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z int
	a = 1
	for idx := 0; idx < 26; idx++ {
		a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z = b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a
	}
	return a*1 + b*2 + c*3 + d*4 + e*5 + f*6 + g*7 + h*8 + i*9 + j*10 + k*11 + l*12 + m*13 + n*14 + o*15 + p*16 + q*17 + r*18 + s*19 + t*20 + u*21 + v*22 + w*23 + x*24 + y*25 + z*26
}

func phiOverwriteBig(t *testing.T) {
	want := 1
	got := phiOverwriteBig_ssa()
	if got != want {
		t.Errorf("phiOverwriteBig_ssa()= %d, got %d", want, got)
	}
}

func TestRegalloc(t *testing.T) {
	phiOverwrite(t)
	phiOverwriteBig(t)
}
