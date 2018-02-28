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

func TestLargeTanNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	large := float64(100000 * Pi)
	for i := 0; i < len(vf); i++ {
		f1 := tanLarge[i]
		f2 := TanNovec(vf[i] + large)
		if !close(f1, f2) {
			t.Errorf("Tan(%g) = %g, want %g", vf[i]+large, f2, f1)
		}
	}
}

func TestTanNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := TanNovec(vf[i]); !veryclose(tan[i], f) {
			t.Errorf("Tan(%g) = %g, want %g", vf[i], f, tan[i])
		}
	}
	// same special cases as Sin
	for i := 0; i < len(vfsinSC); i++ {
		if f := TanNovec(vfsinSC[i]); !alike(sinSC[i], f) {
			t.Errorf("Tan(%g) = %g, want %g", vfsinSC[i], f, sinSC[i])
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

func TestLog1pNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 100
		if f := Log1pNovec(a); !veryclose(log1p[i], f) {
			t.Errorf("Log1p(%g) = %g, want %g", a, f, log1p[i])
		}
	}
	a := 9.0
	if f := Log1pNovec(a); f != Ln10 {
		t.Errorf("Log1p(%g) = %g, want %g", a, f, Ln10)
	}
	for i := 0; i < len(vflogSC); i++ {
		if f := Log1pNovec(vflog1pSC[i]); !alike(log1pSC[i], f) {
			t.Errorf("Log1p(%g) = %g, want %g", vflog1pSC[i], f, log1pSC[i])
		}
	}
}

func TestAtanhNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := AtanhNovec(a); !veryclose(atanh[i], f) {
			t.Errorf("Atanh(%g) = %g, want %g", a, f, atanh[i])
		}
	}
	for i := 0; i < len(vfatanhSC); i++ {
		if f := AtanhNovec(vfatanhSC[i]); !alike(atanhSC[i], f) {
			t.Errorf("Atanh(%g) = %g, want %g", vfatanhSC[i], f, atanhSC[i])
		}
	}
}

func TestAcosNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := AcosNovec(a); !close(acos[i], f) {
			t.Errorf("Acos(%g) = %g, want %g", a, f, acos[i])
		}
	}
	for i := 0; i < len(vfacosSC); i++ {
		if f := AcosNovec(vfacosSC[i]); !alike(acosSC[i], f) {
			t.Errorf("Acos(%g) = %g, want %g", vfacosSC[i], f, acosSC[i])
		}
	}
}

func TestAsinNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := AsinNovec(a); !veryclose(asin[i], f) {
			t.Errorf("Asin(%g) = %g, want %g", a, f, asin[i])
		}
	}
	for i := 0; i < len(vfasinSC); i++ {
		if f := AsinNovec(vfasinSC[i]); !alike(asinSC[i], f) {
			t.Errorf("Asin(%g) = %g, want %g", vfasinSC[i], f, asinSC[i])
		}
	}
}

func TestAcoshNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := 1 + Abs(vf[i])
		if f := AcoshNovec(a); !veryclose(acosh[i], f) {
			t.Errorf("Acosh(%g) = %g, want %g", a, f, acosh[i])
		}
	}
	for i := 0; i < len(vfacoshSC); i++ {
		if f := AcoshNovec(vfacoshSC[i]); !alike(acoshSC[i], f) {
			t.Errorf("Acosh(%g) = %g, want %g", vfacoshSC[i], f, acoshSC[i])
		}
	}
}

func TestAsinhNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := AsinhNovec(vf[i]); !veryclose(asinh[i], f) {
			t.Errorf("Asinh(%g) = %g, want %g", vf[i], f, asinh[i])
		}
	}
	for i := 0; i < len(vfasinhSC); i++ {
		if f := AsinhNovec(vfasinhSC[i]); !alike(asinhSC[i], f) {
			t.Errorf("Asinh(%g) = %g, want %g", vfasinhSC[i], f, asinhSC[i])
		}
	}
}

func TestErfNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := ErfNovec(a); !veryclose(erf[i], f) {
			t.Errorf("Erf(%g) = %g, want %g", a, f, erf[i])
		}
	}
	for i := 0; i < len(vferfSC); i++ {
		if f := ErfNovec(vferfSC[i]); !alike(erfSC[i], f) {
			t.Errorf("Erf(%g) = %g, want %g", vferfSC[i], f, erfSC[i])
		}
	}
}

func TestErfcNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 10
		if f := ErfcNovec(a); !veryclose(erfc[i], f) {
			t.Errorf("Erfc(%g) = %g, want %g", a, f, erfc[i])
		}
	}
	for i := 0; i < len(vferfcSC); i++ {
		if f := ErfcNovec(vferfcSC[i]); !alike(erfcSC[i], f) {
			t.Errorf("Erfc(%g) = %g, want %g", vferfcSC[i], f, erfcSC[i])
		}
	}
}

func TestAtanNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := AtanNovec(vf[i]); !veryclose(atan[i], f) {
			t.Errorf("Atan(%g) = %g, want %g", vf[i], f, atan[i])
		}
	}
	for i := 0; i < len(vfatanSC); i++ {
		if f := AtanNovec(vfatanSC[i]); !alike(atanSC[i], f) {
			t.Errorf("Atan(%g) = %g, want %g", vfatanSC[i], f, atanSC[i])
		}
	}
}

func TestAtan2Novec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := Atan2Novec(10, vf[i]); !veryclose(atan2[i], f) {
			t.Errorf("Atan2(10, %g) = %g, want %g", vf[i], f, atan2[i])
		}
	}
	for i := 0; i < len(vfatan2SC); i++ {
		if f := Atan2Novec(vfatan2SC[i][0], vfatan2SC[i][1]); !alike(atan2SC[i], f) {
			t.Errorf("Atan2(%g, %g) = %g, want %g", vfatan2SC[i][0], vfatan2SC[i][1], f, atan2SC[i])
		}
	}
}

func TestCbrtNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := CbrtNovec(vf[i]); !veryclose(cbrt[i], f) {
			t.Errorf("Cbrt(%g) = %g, want %g", vf[i], f, cbrt[i])
		}
	}
	for i := 0; i < len(vfcbrtSC); i++ {
		if f := CbrtNovec(vfcbrtSC[i]); !alike(cbrtSC[i], f) {
			t.Errorf("Cbrt(%g) = %g, want %g", vfcbrtSC[i], f, cbrtSC[i])
		}
	}
}

func TestLogNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := Abs(vf[i])
		if f := LogNovec(a); log[i] != f {
			t.Errorf("Log(%g) = %g, want %g", a, f, log[i])
		}
	}
	if f := LogNovec(10); f != Ln10 {
		t.Errorf("Log(%g) = %g, want %g", 10.0, f, Ln10)
	}
	for i := 0; i < len(vflogSC); i++ {
		if f := LogNovec(vflogSC[i]); !alike(logSC[i], f) {
			t.Errorf("Log(%g) = %g, want %g", vflogSC[i], f, logSC[i])
		}
	}
}

func TestExpNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	testExpNovec(t, Exp, "Exp")
	testExpNovec(t, ExpGo, "ExpGo")
}

func testExpNovec(t *testing.T, Exp func(float64) float64, name string) {
	for i := 0; i < len(vf); i++ {
		if f := ExpNovec(vf[i]); !veryclose(exp[i], f) {
			t.Errorf("%s(%g) = %g, want %g", name, vf[i], f, exp[i])
		}
	}
	for i := 0; i < len(vfexpSC); i++ {
		if f := ExpNovec(vfexpSC[i]); !alike(expSC[i], f) {
			t.Errorf("%s(%g) = %g, want %g", name, vfexpSC[i], f, expSC[i])
		}
	}
}

func TestExpm1Novec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		a := vf[i] / 100
		if f := Expm1Novec(a); !veryclose(expm1[i], f) {
			t.Errorf("Expm1(%g) = %g, want %g", a, f, expm1[i])
		}
	}
	for i := 0; i < len(vf); i++ {
		a := vf[i] * 10
		if f := Expm1Novec(a); !close(expm1Large[i], f) {
			t.Errorf("Expm1(%g) = %g, want %g", a, f, expm1Large[i])
		}
	}
	for i := 0; i < len(vfexpm1SC); i++ {
		if f := Expm1Novec(vfexpm1SC[i]); !alike(expm1SC[i], f) {
			t.Errorf("Expm1(%g) = %g, want %g", vfexpm1SC[i], f, expm1SC[i])
		}
	}
}

func TestPowNovec(t *testing.T) {
	if !HasVX {
		t.Skipf("no vector support")
	}
	for i := 0; i < len(vf); i++ {
		if f := PowNovec(10, vf[i]); !close(pow[i], f) {
			t.Errorf("Pow(10, %g) = %g, want %g", vf[i], f, pow[i])
		}
	}
	for i := 0; i < len(vfpowSC); i++ {
		if f := PowNovec(vfpowSC[i][0], vfpowSC[i][1]); !alike(powSC[i], f) {
			t.Errorf("Pow(%g, %g) = %g, want %g", vfpowSC[i][0], vfpowSC[i][1], f, powSC[i])
		}
	}
}
