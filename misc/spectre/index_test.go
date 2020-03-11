// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package spectre

import "testing"

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("index did not panic")
		}
	}()
	f()
}

var (
	Zero  = 0
	One   = 1
	Two   = 2
	Three = 3
	Four  = 4
	Five  = 5
)

func TestIndex(t *testing.T) {
	xs := "hello"
	xi := []int{10, 20, 30, 40, 50}
	xf := []float64{10, 20, 30, 40, 50}

	xs = xs[Zero:Five]
	xi = xi[Zero:Five]
	xf = xf[Zero:Five]

	if xs[Four] != 'o' {
		t.Errorf("xs[4] = %q, want %q", xs[Four], 'o')
	}
	if xi[Four] != 50 {
		t.Errorf("xi[4] = %d, want 50", xi[Four])
	}
	if xf[Four] != 50 {
		t.Errorf("xf[4] = %v, want 50", xf[Four])
	}

	xs1 := xs[One:]
	xi1 := xi[One:]
	xf1 := xf[One:]

	if xs1[Three] != 'o' {
		t.Errorf("xs1[3] = %q, want %q", xs1[Three], 'o')
	}
	if xi1[Three] != 50 {
		t.Errorf("xi1[3] = %d, want 50", xi1[Three])
	}
	if xf1[Three] != 50 {
		t.Errorf("xf1[3] = %v, want 50", xf1[Three])
	}
}
