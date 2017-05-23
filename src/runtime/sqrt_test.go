// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A copy of Sqrt tests from the math package to test the
// purely integer arithmetic implementation in sqrt.go.

package runtime_test

import (
	"math"
	"runtime"
	"testing"
)

func SqrtRT(x float64) float64 {
	return math.Float64frombits(runtime.Sqrt(math.Float64bits(x)))
}

func TestSqrt(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := math.Abs(vf[i])
		if f := SqrtRT(a); sqrt[i] != f {
			t.Errorf("Sqrt(%g) = %g, want %g", a, f, sqrt[i])
		}
	}
	for i := 0; i < len(vfsqrtSC); i++ {
		if f := SqrtRT(vfsqrtSC[i]); !alike(sqrtSC[i], f) {
			t.Errorf("Sqrt(%g) = %g, want %g", vfsqrtSC[i], f, sqrtSC[i])
		}
	}
}

func alike(a, b float64) bool {
	switch {
	case math.IsNaN(a) && math.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(a) == math.Signbit(b)
	}
	return false
}

var vf = []float64{
	4.9790119248836735e+00,
	7.7388724745781045e+00,
	-2.7688005719200159e-01,
	-5.0106036182710749e+00,
	9.6362937071984173e+00,
	2.9263772392439646e+00,
	5.2290834314593066e+00,
	2.7279399104360102e+00,
	1.8253080916808550e+00,
	-8.6859247685756013e+00,
}

var sqrt = []float64{
	2.2313699659365484748756904e+00,
	2.7818829009464263511285458e+00,
	5.2619393496314796848143251e-01,
	2.2384377628763938724244104e+00,
	3.1042380236055381099288487e+00,
	1.7106657298385224403917771e+00,
	2.286718922705479046148059e+00,
	1.6516476350711159636222979e+00,
	1.3510396336454586262419247e+00,
	2.9471892997524949215723329e+00,
}

var vfsqrtSC = []float64{
	math.Inf(-1),
	-math.Pi,
	math.Copysign(0, -1),
	0,
	math.Inf(1),
	math.NaN(),
	math.Float64frombits(2),
}
var sqrtSC = []float64{
	math.NaN(),
	math.NaN(),
	math.Copysign(0, -1),
	0,
	math.Inf(1),
	math.NaN(),
	3.1434555694052576e-162,
}
