// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func
close(a, b double) bool
{
	if a == 0 {
		if b == 0 {
			return true;
		}
		return false;
	}
	d := a-b;
	if d < 0 {
		d = -d;
	}
	e := a;
	if e < 0 {
		e = -e;
	}
	if e*1.0e-14 > d {
		return true;
	}
	return false;
}

func main() int {

	if !close(0., 0.) { print "0. is ", 0., " should be ", 0., "\n"; return 1; }
	if !close(+10., +10.) { print "+10. is ", +10., " should be ", +10., "\n"; return 1; }
	if !close(-210., -210.) { print "-210. is ", -210., " should be ", -210., "\n"; return 1; }

	if !close(.0, .0) { print ".0 is ", .0, " should be ", .0, "\n"; return 1; }
	if !close(+.01, +.01) { print "+.01 is ", +.01, " should be ", +.01, "\n"; return 1; }
	if !close(-.012, -.012) { print "-.012 is ", -.012, " should be ", -.012, "\n"; return 1; }

	if !close(0.0, 0.0) { print "0.0 is ", 0.0, " should be ", 0.0, "\n"; return 1; }
	if !close(+10.01, +10.01) { print "+10.01 is ", +10.01, " should be ", +10.01, "\n"; return 1; }
	if !close(-210.012, -210.012) { print "-210.012 is ", -210.012, " should be ", -210.012, "\n"; return 1; }

	if !close(0E+1, 0E+1) { print "0E+1 is ", 0E+1, " should be ", 0E+1, "\n"; return 1; }
	if !close(+10e2, +10e2) { print "+10e2 is ", +10e2, " should be ", +10e2, "\n"; return 1; }
	if !close(-210e3, -210e3) { print "-210e3 is ", -210e3, " should be ", -210e3, "\n"; return 1; }

	if !close(0E-1, 0E-1) { print "0E-1 is ", 0E-1, " should be ", 0E-1, "\n"; return 1; }
	if !close(+0e23, +0e23) { print "+0e23 is ", +0e23, " should be ", +0e23, "\n"; return 1; }
	if !close(-0e345, -0e345) { print "-0e345 is ", -0e345, " should be ", -0e345, "\n"; return 1; }

	if !close(0E1, 0E1) { print "0E1 is ", 0E1, " should be ", 0E1, "\n"; return 1; }
	if !close(+10e23, +10e23) { print "+10e23 is ", +10e23, " should be ", +10e23, "\n"; return 1; }
//	if !close(-210e345, -210e345) { print "-210e345 is ", -210e345, " should be ", -210e345, "\n"; return 1; }

	if !close(0.E1, 0.E1) { print "0.E1 is ", 0.E1, " should be ", 0.E1, "\n"; return 1; }
	if !close(+10.e+2, +10.e+2) { print "+10.e+2 is ", +10.e+2, " should be ", +10.e+2, "\n"; return 1; }
	if !close(-210.e-3, -210.e-3) { print "-210.e-3 is ", -210.e-3, " should be ", -210.e-3, "\n"; return 1; }

	if !close(.0E1, .0E1) { print ".0E1 is ", .0E1, " should be ", .0E1, "\n"; return 1; }
	if !close(+.01e2, +.01e2) { print "+.01e2 is ", +.01e2, " should be ", +.01e2, "\n"; return 1; }
	if !close(-.012e3, -.012e3) { print "-.012e3 is ", -.012e3, " should be ", -.012e3, "\n"; return 1; }

	if !close(0.0E1, 0.0E1) { print "0.0E1 is ", 0.0E1, " should be ", 0.0E1, "\n"; return 1; }
	if !close(+10.01e2, +10.01e2) { print "+10.01e2 is ", +10.01e2, " should be ", +10.01e2, "\n"; return 1; }
	if !close(-210.012e3, -210.012e3) { print "-210.012e3 is ", -210.012e3, " should be ", -210.012e3, "\n"; return 1; }

	if !close(0.E+12, 0.E+12) { print "0.E+12 is ", 0.E+12, " should be ", 0.E+12, "\n"; return 1; }
	if !close(+10.e23, +10.e23) { print "+10.e23 is ", +10.e23, " should be ", +10.e23, "\n"; return 1; }
	if !close(-210.e34, -210.e34) { print "-210.e34 is ", -210.e34, " should be ", -210.e34, "\n"; return 1; }

	if !close(.0E-12, .0E-12) { print ".0E-12 is ", .0E-12, " should be ", .0E-12, "\n"; return 1; }
	if !close(+.01e23, +.01e23) { print "+.01e23 is ", +.01e23, " should be ", +.01e23, "\n"; return 1; }
	if !close(-.012e34, -.012e34) { print "-.012e34 is ", -.012e34, " should be ", -.012e34, "\n"; return 1; }

	if !close(0.0E12, 0.0E12) { print "0.0E12 is ", 0.0E12, " should be ", 0.0E12, "\n"; return 1; }
	if !close(+10.01e23, +10.01e23) { print "+10.01e23 is ", +10.01e23, " should be ", +10.01e23, "\n"; return 1; }
	if !close(-210.012e34, -210.012e34) { print "-210.012e34 is ", -210.012e34, " should be ", -210.012e34, "\n"; return 1; }

	if !close(0.E123, 0.E123) { print "0.E123 is ", 0.E123, " should be ", 0.E123, "\n"; return 1; }
	if !close(+10.e+234, +10.e+234) { print "+10.e+234 is ", +10.e+234, " should be ", +10.e+234, "\n"; return 1; }
//	if !close(-210.e-345, -210.e-345) { print "-210.e-345 is ", -210.e-345, " should be ", -210.e-345, "\n"; return 1; }

	if !close(.0E123, .0E123) { print ".0E123 is ", .0E123, " should be ", .0E123, "\n"; return 1; }
//	if !close(+.01e234, +.01e234) { print "+.01e234 is ", +.01e234, " should be ", +.01e234, "\n"; return 1; }
//	if !close(-.012e345, -.012e345) { print "-.012e345 is ", -.012e345, " should be ", -.012e345, "\n"; return 1; }

	if !close(0.0E123, 0.0E123) { print "0.0E123 is ", 0.0E123, " should be ", 0.0E123, "\n"; return 1; }
//	if !close(+10.01e234, +10.01e234) { print "+10.01e234 is ", +10.01e234, " should be ", +10.01e234, "\n"; return 1; }
//	if !close(-210.012e345, -210.012e345) { print "-210.012e345 is ", -210.012e345, " should be ", -210.012e345, "\n"; return 1; }
}
