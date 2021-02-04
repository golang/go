// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import "testing"

var f52want float64 = 1.0 / (1 << 52)
var f53want float64 = 1.0 / (1 << 53)

func TestTruncFlt(t *testing.T) {
	const f52 = 1 + 1.0/(1<<52)
	const f53 = 1 + 1.0/(1<<53)

	if got := f52 - 1; got != f52want {
		t.Errorf("f52-1 = %g, want %g", got, f52want)
	}
	if got := float64(f52) - 1; got != f52want {
		t.Errorf("float64(f52)-1 = %g, want %g", got, f52want)
	}
	if got := f53 - 1; got != f53want {
		t.Errorf("f53-1 = %g, want %g", got, f53want)
	}
	if got := float64(f53) - 1; got != 0 {
		t.Errorf("float64(f53)-1 = %g, want 0", got)
	}
}

func TestTruncCmplx(t *testing.T) {
	const r52 = complex(1+1.0/(1<<52), 0)
	const r53 = complex(1+1.0/(1<<53), 0)

	if got := real(r52 - 1); got != f52want {
		t.Errorf("real(r52-1) = %g, want %g", got, f52want)
	}
	if got := real(complex128(r52) - 1); got != f52want {
		t.Errorf("real(complex128(r52)-1) = %g, want %g", got, f52want)
	}
	if got := real(r53 - 1); got != f53want {
		t.Errorf("real(r53-1) = %g, want %g", got, f53want)
	}
	if got := real(complex128(r53) - 1); got != 0 {
		t.Errorf("real(complex128(r53)-1) = %g, want 0", got)
	}

	const i52 = complex(0, 1+1.0/(1<<52))
	const i53 = complex(0, 1+1.0/(1<<53))

	if got := imag(i52 - 1i); got != f52want {
		t.Errorf("imag(i52-1i) = %g, want %g", got, f52want)
	}
	if got := imag(complex128(i52) - 1i); got != f52want {
		t.Errorf("imag(complex128(i52)-1i) = %g, want %g", got, f52want)
	}
	if got := imag(i53 - 1i); got != f53want {
		t.Errorf("imag(i53-1i) = %g, want %g", got, f53want)
	}
	if got := imag(complex128(i53) - 1i); got != 0 {
		t.Errorf("imag(complex128(i53)-1i) = %g, want 0", got)
	}

}
