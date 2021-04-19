// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests whether the non vector routines are working, even when the tests are run on a
// vector-capable machine.
package math_test

import (
	. "math"
	"testing"
)

func TestCosNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := CosNoVec(vf[i]); !veryclose(cos[i], f) {
			t.Errorf("Cos(%g) = %g, want %g", vf[i], f, cos[i])
		}
	}
	for i := 0; i < len(vfcosSC); i++ {
		if f := CosNoVec(vfcosSC[i]); !alike(cosSC[i], f) {
			t.Errorf("Cos(%g) = %g, want %g", vfcosSC[i], f, cosSC[i])
		}
	}
}

func TestCoshNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := CoshNoVec(vf[i]); !close(cosh[i], f) {
			t.Errorf("Cosh(%g) = %g, want %g", vf[i], f, cosh[i])
		}
	}
	for i := 0; i < len(vfcoshSC); i++ {
		if f := CoshNoVec(vfcoshSC[i]); !alike(coshSC[i], f) {
			t.Errorf("Cosh(%g) = %g, want %g", vfcoshSC[i], f, coshSC[i])
		}
	}
}
func TestSinNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := SinNoVec(vf[i]); !veryclose(sin[i], f) {
			t.Errorf("Sin(%g) = %g, want %g", vf[i], f, sin[i])
		}
	}
	for i := 0; i < len(vfsinSC); i++ {
		if f := SinNoVec(vfsinSC[i]); !alike(sinSC[i], f) {
			t.Errorf("Sin(%g) = %g, want %g", vfsinSC[i], f, sinSC[i])
		}
	}
}

func TestSinhNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := SinhNoVec(vf[i]); !close(sinh[i], f) {
			t.Errorf("Sinh(%g) = %g, want %g", vf[i], f, sinh[i])
		}
	}
	for i := 0; i < len(vfsinhSC); i++ {
		if f := SinhNoVec(vfsinhSC[i]); !alike(sinhSC[i], f) {
			t.Errorf("Sinh(%g) = %g, want %g", vfsinhSC[i], f, sinhSC[i])
		}
	}
}

// Check that math functions of high angle values
// return accurate results. [Since (vf[i] + large) - large != vf[i],
// testing for Trig(vf[i] + large) == Trig(vf[i]), where large is
// a multiple of 2*Pi, is misleading.]
func TestLargeCosNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	large := float64(100000 * Pi)
	for i := 0; i < len(vf); i++ {
		f1 := cosLarge[i]
		f2 := CosNoVec(vf[i] + large)
		if !close(f1, f2) {
			t.Errorf("Cos(%g) = %g, want %g", vf[i]+large, f2, f1)
		}
	}
}

func TestLargeSinNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	large := float64(100000 * Pi)
	for i := 0; i < len(vf); i++ {
		f1 := sinLarge[i]
		f2 := SinNoVec(vf[i] + large)
		if !close(f1, f2) {
			t.Errorf("Sin(%g) = %g, want %g", vf[i]+large, f2, f1)
		}
	}
}

func TestTanhNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := TanhNoVec(vf[i]); !veryclose(tanh[i], f) {
			t.Errorf("Tanh(%g) = %g, want %g", vf[i], f, tanh[i])
		}
	}
	for i := 0; i < len(vftanhSC); i++ {
		if f := TanhNoVec(vftanhSC[i]); !alike(tanhSC[i], f) {
			t.Errorf("Tanh(%g) = %g, want %g", vftanhSC[i], f, tanhSC[i])
		}
	}

}

func TestLog10Novec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := Abs(vf[i])
		if f := Log10NoVec(a); !veryclose(log10[i], f) {
			t.Errorf("Log10(%g) = %g, want %g", a, f, log10[i])
		}
	}
	if f := Log10NoVec(E); f != Log10E {
		t.Errorf("Log10(%g) = %g, want %g", E, f, Log10E)
	}
	for i := 0; i < len(vflogSC); i++ {
		if f := Log10NoVec(vflogSC[i]); !alike(logSC[i], f) {
			t.Errorf("Log10(%g) = %g, want %g", vflogSC[i], f, logSC[i])
		}
	}
}
