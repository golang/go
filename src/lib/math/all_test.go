// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import (
	"math";
	"testing";
)

var vf = []float64 (
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
)
var asin = []float64 (
	  5.2117697218417440e-01,
	  8.8495619865825236e-01,
	 -2.7691544662819413e-02,
	 -5.2482360935268932e-01,
	  1.3002662421166553e+00,
	  2.9698415875871901e-01,
	  5.5025938468083364e-01,
	  2.7629597861677200e-01,
	  1.8355989225745148e-01,
	 -1.0523547536021498e+00,
)
var atan = []float64 (
	  1.3725902621296217e+00,
	  1.4422906096452980e+00,
	 -2.7011324359471755e-01,
	 -1.3738077684543379e+00,
	  1.4673921193587666e+00,
	  1.2415173565870167e+00,
	  1.3818396865615167e+00,
	  1.2194305844639670e+00,
	  1.0696031952318783e+00,
	 -1.4561721938838085e+00,
)
var exp = []float64 (
	  1.4533071302642137e+02,
	  2.2958822575694450e+03,
	  7.5814542574851666e-01,
	  6.6668778421791010e-03,
	  1.5310493273896035e+04,
	  1.8659907517999329e+01,
	  1.8662167355098713e+02,
	  1.5301332413189379e+01,
	  6.2047063430646876e+00,
	  1.6894712385826522e-04,
)
var floor = []float64 (
	  4.0000000000000000e+00,
	  7.0000000000000000e+00,
	 -1.0000000000000000e+00,
	 -6.0000000000000000e+00,
	  9.0000000000000000e+00,
	  2.0000000000000000e+00,
	  5.0000000000000000e+00,
	  2.0000000000000000e+00,
	  1.0000000000000000e+00,
	 -9.0000000000000000e+00,
)
var log = []float64 (
	  1.6052314626930630e+00,
	  2.0462560018708768e+00,
	 -1.2841708730962657e+00,
	  1.6115563905281544e+00,
	  2.2655365644872018e+00,
	  1.0737652208918380e+00,
	  1.6542360106073545e+00,
	  1.0035467127723465e+00,
	  6.0174879014578053e-01,
	  2.1617038728473527e+00,
)
var pow = []float64 (
	  9.5282232631648415e+04,
	  5.4811599352999900e+07,
	  5.2859121715894400e-01,
	  9.7587991957286472e-06,
	  4.3280643293460450e+09,
	  8.4406761805034551e+02,
	  1.6946633276191194e+05,
	  5.3449040147551940e+02,
	  6.6881821384514159e+01,
	  2.0609869004248744e-09,
)
var sin = []float64 (
	 -9.6466616586009283e-01,
	  9.9338225271646543e-01,
	 -2.7335587039794395e-01,
	  9.5586257685042800e-01,
	 -2.0994210667799692e-01,
	  2.1355787807998605e-01,
	 -8.6945689711673619e-01,
	  4.0195666811555783e-01,
	  9.6778633541688000e-01,
	 -6.7344058690503452e-01,
)
var sinh = []float64 (
	  7.2661916084208533e+01,
	  1.1479409110035194e+03,
	 -2.8043136512812520e-01,
	 -7.4994290911815868e+01,
	  7.6552466042906761e+03,
	  9.3031583421672010e+00,
	  9.3308157558281088e+01,
	  7.6179893137269143e+00,
	  3.0217691805496156e+00,
	 -2.9595057572444951e+03,
)
var sqrt = []float64 (
	  2.2313699659365484e+00,
	  2.7818829009464263e+00,
	  5.2619393496314792e-01,
	  2.2384377628763938e+00,
	  3.1042380236055380e+00,
	  1.7106657298385224e+00,
	  2.2867189227054791e+00,
	  1.6516476350711160e+00,
	  1.3510396336454586e+00,
	  2.9471892997524950e+00,
)
var tan = []float64 (
	 -3.6613165650402277e+00,
	  8.6490023264859754e+00,
	 -2.8417941955033615e-01,
	  3.2532901859747287e+00,
	  2.1472756403802937e-01,
	 -2.1860091071106700e-01,
	 -1.7600028178723679e+00,
	 -4.3898089147528178e-01,
	 -3.8438855602011305e+00,
	  9.1098879337768517e-01,
)
var tanh = []float64 (
	  9.9990531206936328e-01,
	  9.9999962057085307e-01,
	 -2.7001505097318680e-01,
	 -9.9991110943061700e-01,
	  9.9999999146798441e-01,
	  9.9427249436125233e-01,
	  9.9994257600983156e-01,
	  9.9149409509772863e-01,
	  9.4936501296239700e-01,
	 -9.9999994291374019e-01,
)

func tolerance(a,b,e float64) bool {
	d := a-b;
	if d < 0 {
		d = -d;
	}

	if a != 0 {
		e = e*a;
		if e < 0 {
			e = -e;
		}
	}
	return d < e;
}
func close(a,b float64) bool {
	return tolerance(a, b, 1e-14);
}
func veryclose(a,b float64) bool {
	return tolerance(a, b, 4e-16);
}

func TestAsin(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Asin(vf[i]/10); !veryclose(asin[i], f) {
			t.Errorf("math.Asin(%g) = %g, want %g\n", vf[i]/10, f, asin[i]);
		}
	}
}

func TestAtan(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Atan(vf[i]); !veryclose(atan[i], f) {
			t.Errorf("math.Atan(%g) = %g, want %g\n", vf[i], f, atan[i]);
		}
	}
}

func TestExp(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Exp(vf[i]); !veryclose(exp[i], f) {
			t.Errorf("math.Exp(%g) = %g, want %g\n", vf[i], f, exp[i]);
		}
	}
}

func TestFloor(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Floor(vf[i]); floor[i] != f {
			t.Errorf("math.Floor(%g) = %g, want %g\n", vf[i], f, floor[i]);
		}
	}
}

func TestLog(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := math.Fabs(vf[i]);
		if f := math.Log(a); log[i] != f {
			t.Errorf("math.Log(%g) = %g, want %g\n", a, f, log[i]);
		}
	}
	if f := math.Log(10); f != math.Ln10 {
		t.Errorf("math.Log(%g) = %g, want %g\n", 10, f, math.Ln10);
	}
}

func TestPow(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Pow(10, vf[i]); !close(pow[i], f) {
			t.Errorf("math.Pow(10, %.17g) = %.17g, want %.17g\n", vf[i], f, pow[i]);
		}
	}
}

func TestSin(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Sin(vf[i]); !close(sin[i], f) {
			t.Errorf("math.Sin(%g) = %g, want %g\n", vf[i], f, sin[i]);
		}
	}
}

func TestSinh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Sinh(vf[i]); !veryclose(sinh[i], f) {
			t.Errorf("math.Sinh(%g) = %g, want %g\n", vf[i], f, sinh[i]);
		}
	}
}

func TestSqrt(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := math.Fabs(vf[i]);
		if f := math.Sqrt(a); !veryclose(sqrt[i], f) {
			t.Errorf("math.Sqrt(%g) = %g, want %g\n", a, f, floor[i]);
		}
	}
}

func TestTan(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Tan(vf[i]); !close(tan[i], f) {
			t.Errorf("math.Tan(%g) = %g, want %g\n", vf[i], f, tan[i]);
		}
	}
}

func TestTanh(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		if f := math.Tanh(vf[i]); !veryclose(tanh[i], f) {
			t.Errorf("math.Tanh(%g) = %g, want %g\n", vf[i], f, tanh[i]);
		}
	}
}

func TestHypot(t *testing.T) {
	for i := 0; i < len(vf); i++ {
		a := math.Fabs(tanh[i]*math.Sqrt(2));
		if f := math.Hypot(tanh[i], tanh[i]); !veryclose(a, f) {
			t.Errorf("math.Hypot(%g, %g) = %g, want %g\n", tanh[i], tanh[i], f, a);
		}
	}
}
