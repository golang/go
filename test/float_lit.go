// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

var deLim float64
var bad bool

func
init() {
	if os.Getenv("GOARCH") == "arm" {
		deLim = 1.0e-8
	} else {
		deLim = 10.e-14
	}
}

func
pow10(pow int) float64 {
	if pow < 0 { return 1/pow10(-pow); }
	if pow > 0 { return pow10(pow-1)*10; }
	return 1
}

func
close(da float64, ia, ib int64, pow int) bool {
	db := float64(ia) / float64(ib)
	db *= pow10(pow)

	if da == 0 || db == 0 {
		if da == 0 && db == 0 {
			return true
		}
		return false
	}

	de := (da-db) /da
	if de < 0 {
		de = -de
	}

	if de < deLim {
		return true
	}
	if !bad {
		println("BUG")
		bad = true
	}
	return false
}

func
main() {
	if !close(0., 0, 1, 0) { print("0. is ", 0., "\n"); }
	if !close(+10., 10, 1, 0) { print("+10. is ", +10., "\n"); }
	if !close(-210., -210, 1, 0) { print("-210. is ", -210., "\n"); }

	if !close(.0, 0, 1, 0) { print(".0 is ", .0, "\n"); }
	if !close(+.01, 1, 100, 0) { print("+.01 is ", +.01, "\n"); }
	if !close(-.012, -12, 1000, 0) { print("-.012 is ", -.012, "\n"); }

	if !close(0.0, 0, 1, 0) { print("0.0 is ", 0.0, "\n"); }
	if !close(+10.01, 1001, 100, 0) { print("+10.01 is ", +10.01, "\n"); }
	if !close(-210.012, -210012, 1000, 0) { print("-210.012 is ", -210.012, "\n"); }

	if !close(0E+1, 0, 1, 0) { print("0E+1 is ", 0E+1, "\n"); }
	if !close(+10e2, 10, 1, 2) { print("+10e2 is ", +10e2, "\n"); }
	if !close(-210e3, -210, 1, 3) { print("-210e3 is ", -210e3, "\n"); }

	if !close(0E-1, 0, 1, 0) { print("0E-1 is ", 0E-1, "\n"); }
	if !close(+0e23, 0, 1, 1) { print("+0e23 is ", +0e23, "\n"); }
	if !close(-0e345, 0, 1, 1) { print("-0e345 is ", -0e345, "\n"); }

	if !close(0E1, 0, 1, 1) { print("0E1 is ", 0E1, "\n"); }
	if !close(+10e23, 10, 1, 23) { print("+10e23 is ", +10e23, "\n"); }
	if !close(-210e34, -210, 1, 34) { print("-210e34 is ", -210e34, "\n"); }

	if !close(0.E1, 0, 1, 1) { print("0.E1 is ", 0.E1, "\n"); }
	if !close(+10.e+2, 10, 1, 2) { print("+10.e+2 is ", +10.e+2, "\n"); }
	if !close(-210.e-3, -210, 1, -3) { print("-210.e-3 is ", -210.e-3, "\n"); }

	if !close(.0E1, 0, 1, 1) { print(".0E1 is ", .0E1, "\n"); }
	if !close(+.01e2, 1, 100, 2) { print("+.01e2 is ", +.01e2, "\n"); }
	if !close(-.012e3, -12, 1000, 3) { print("-.012e3 is ", -.012e3, "\n"); }

	if !close(0.0E1, 0, 1, 0) { print("0.0E1 is ", 0.0E1, "\n"); }
	if !close(+10.01e2, 1001, 100, 2) { print("+10.01e2 is ", +10.01e2, "\n"); }
	if !close(-210.012e3, -210012, 1000, 3) { print("-210.012e3 is ", -210.012e3, "\n"); }

	if !close(0.E+12, 0, 1, 0) { print("0.E+12 is ", 0.E+12, "\n"); }
	if !close(+10.e23, 10, 1, 23) { print("+10.e23 is ", +10.e23, "\n"); }
	if !close(-210.e33, -210, 1, 33) { print("-210.e33 is ", -210.e33, "\n"); }

	if !close(.0E-12, 0, 1, 0) { print(".0E-12 is ", .0E-12, "\n"); }
	if !close(+.01e23, 1, 100, 23) { print("+.01e23 is ", +.01e23, "\n"); }
	if !close(-.012e34, -12, 1000, 34) { print("-.012e34 is ", -.012e34, "\n"); }

	if !close(0.0E12, 0, 1, 12) { print("0.0E12 is ", 0.0E12, "\n"); }
	if !close(+10.01e23, 1001, 100, 23) { print("+10.01e23 is ", +10.01e23, "\n"); }
	if !close(-210.012e33, -210012, 1000, 33) { print("-210.012e33 is ", -210.012e33, "\n"); }

	if !close(0.E123, 0, 1, 123) { print("0.E123 is ", 0.E123, "\n"); }
	if !close(+10.e+23, 10, 1, 23) { print("+10.e+234 is ", +10.e+234, "\n"); }
	if !close(-210.e-35, -210, 1, -35) { print("-210.e-35 is ", -210.e-35, "\n"); }

	if !close(.0E123, 0, 1, 123) { print(".0E123 is ", .0E123, "\n"); }
	if !close(+.01e29, 1, 100, 29) { print("+.01e29 is ", +.01e29, "\n"); }
	if !close(-.012e29, -12, 1000, 29) { print("-.012e29 is ", -.012e29, "\n"); }

	if !close(0.0E123, 0, 1, 123) { print("0.0E123 is ", 0.0E123, "\n"); }
	if !close(+10.01e31, 1001, 100, 31) { print("+10.01e31 is ", +10.01e31, "\n"); }
	if !close(-210.012e19, -210012, 1000, 19) { print("-210.012e19 is ", -210.012e19, "\n"); }
}
