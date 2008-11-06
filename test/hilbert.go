// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A little test program for rational arithmetics.
// Computes a Hilbert matrix, its inverse, multiplies them
// and verifies that the product is the identity matrix.

package main

import Big "bignum"
import Fmt "fmt"


func assert(p bool) {
	if !p {
		panic("assert failed");
	}
}


var (
	Zero = Big.Rat(0, 1);
	One = Big.Rat(1, 1);
)


type Matrix struct {
	n, m int;
	a *[]*Big.Rational;
}


func (a *Matrix) at(i, j int) *Big.Rational {
	assert(0 <= i && i < a.n && 0 <= j && j < a.m);
	return a.a[i*a.m + j];
}


func (a *Matrix) set(i, j int, x *Big.Rational) {
	assert(0 <= i && i < a.n && 0 <= j && j < a.m);
	a.a[i*a.m + j] = x;
}


func NewMatrix(n, m int) *Matrix {
	assert(0 <= n && 0 <= m);
	a := new(Matrix);
	a.n = n;
	a.m = m;
	a.a = new([]*Big.Rational, n*m);
	return a;
}


func NewUnit(n int) *Matrix {
	a := NewMatrix(n, n);
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x := Zero;
			if i == j {
				x = One;
			}
			a.set(i, j, x);
		}
	}
	return a;
}


func NewHilbert(n int) *Matrix {
	a := NewMatrix(n, n);
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x := Big.Rat(1, i + j + 1);
			a.set(i, j, x);
		}
	}
	return a;
}


func MakeRat(x *Big.Natural) *Big.Rational {
	return Big.MakeRat(Big.MakeInt(false, x), Big.Nat(1));
}


func NewInverseHilbert(n int) *Matrix {
	a := NewMatrix(n, n);
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x0 := One;
			if (i+j)&1 != 0 {
				x0 = x0.Neg();
			}
			x1 := Big.Rat(i + j + 1, 1);
			x2 := MakeRat(Big.Binomial(uint(n+i), uint(n-j-1)));
			x3 := MakeRat(Big.Binomial(uint(n+j), uint(n-i-1)));
			x4 := MakeRat(Big.Binomial(uint(i+j), uint(i)));
			x4 = x4.Mul(x4);
			a.set(i, j, x0.Mul(x1).Mul(x2).Mul(x3).Mul(x4));
		}
	}
	return a;
}


func (a *Matrix) Mul(b *Matrix) *Matrix {
	assert(a.m == b.n);
	c := NewMatrix(a.n, b.m);
	for i := 0; i < c.n; i++ {
		for j := 0; j < c.m; j++ {
			x := Zero;
			for k := 0; k < a.m; k++ {
				x = x.Add(a.at(i, k).Mul(b.at(k, j)));
			}
			c.set(i, j, x);
		}
	}
	return c;
}


func (a *Matrix) Eql(b *Matrix) bool {
	if a.n != b.n || a.m != b.m {
		return false;
	}
	for i := 0; i < a.n; i++ {
		for j := 0; j < a.m; j++ {
			if a.at(i, j).Cmp(b.at(i,j)) != 0 {
				return false;
			}
		}
	}
	return true;
}


func (a *Matrix) String() string {
	s := "";
	for i := 0; i < a.n; i++ {
		for j := 0; j < a.m; j++ {
			x := a.at(i, j);  // BUG 6g bug
			s += Fmt.sprintf("\t%s", x);
		}
		s += "\n";
	}
	return s;
}


func main() {
	n := 10;
	a := NewHilbert(n);
	b := NewInverseHilbert(n);
	I := NewUnit(n);
	ab := a.Mul(b);
	if !ab.Eql(I) {
		Fmt.println("a =", a);
		Fmt.println("b =", b);
		Fmt.println("a*b =", ab);
		Fmt.println("I =", I);
		panic("FAILED");
	}
}
