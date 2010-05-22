// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A little test program and benchmark for rational arithmetics.
// Computes a Hilbert matrix, its inverse, multiplies them
// and verifies that the product is the identity matrix.

package big

import (
	"fmt"
	"testing"
)


type matrix struct {
	n, m int
	a    []*Rat
}


func (a *matrix) at(i, j int) *Rat {
	if !(0 <= i && i < a.n && 0 <= j && j < a.m) {
		panic("index out of range")
	}
	return a.a[i*a.m+j]
}


func (a *matrix) set(i, j int, x *Rat) {
	if !(0 <= i && i < a.n && 0 <= j && j < a.m) {
		panic("index out of range")
	}
	a.a[i*a.m+j] = x
}


func newMatrix(n, m int) *matrix {
	if !(0 <= n && 0 <= m) {
		panic("illegal matrix")
	}
	a := new(matrix)
	a.n = n
	a.m = m
	a.a = make([]*Rat, n*m)
	return a
}


func newUnit(n int) *matrix {
	a := newMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x := NewRat(0, 1)
			if i == j {
				x.SetInt64(1)
			}
			a.set(i, j, x)
		}
	}
	return a
}


func newHilbert(n int) *matrix {
	a := newMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			a.set(i, j, NewRat(1, int64(i+j+1)))
		}
	}
	return a
}


func newInverseHilbert(n int) *matrix {
	a := newMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			x1 := new(Rat).SetInt64(int64(i + j + 1))
			x2 := new(Rat).SetInt(new(Int).Binomial(int64(n+i), int64(n-j-1)))
			x3 := new(Rat).SetInt(new(Int).Binomial(int64(n+j), int64(n-i-1)))
			x4 := new(Rat).SetInt(new(Int).Binomial(int64(i+j), int64(i)))

			x1.Mul(x1, x2)
			x1.Mul(x1, x3)
			x1.Mul(x1, x4)
			x1.Mul(x1, x4)

			if (i+j)&1 != 0 {
				x1.Neg(x1)
			}

			a.set(i, j, x1)
		}
	}
	return a
}


func (a *matrix) mul(b *matrix) *matrix {
	if a.m != b.n {
		panic("illegal matrix multiply")
	}
	c := newMatrix(a.n, b.m)
	for i := 0; i < c.n; i++ {
		for j := 0; j < c.m; j++ {
			x := NewRat(0, 1)
			for k := 0; k < a.m; k++ {
				x.Add(x, new(Rat).Mul(a.at(i, k), b.at(k, j)))
			}
			c.set(i, j, x)
		}
	}
	return c
}


func (a *matrix) eql(b *matrix) bool {
	if a.n != b.n || a.m != b.m {
		return false
	}
	for i := 0; i < a.n; i++ {
		for j := 0; j < a.m; j++ {
			if a.at(i, j).Cmp(b.at(i, j)) != 0 {
				return false
			}
		}
	}
	return true
}


func (a *matrix) String() string {
	s := ""
	for i := 0; i < a.n; i++ {
		for j := 0; j < a.m; j++ {
			s += fmt.Sprintf("\t%s", a.at(i, j))
		}
		s += "\n"
	}
	return s
}


func doHilbert(t *testing.T, n int) {
	a := newHilbert(n)
	b := newInverseHilbert(n)
	I := newUnit(n)
	ab := a.mul(b)
	if !ab.eql(I) {
		if t == nil {
			panic("Hilbert failed")
		}
		t.Errorf("a   = %s\n", a)
		t.Errorf("b   = %s\n", b)
		t.Errorf("a*b = %s\n", ab)
		t.Errorf("I   = %s\n", I)
	}
}


func TestHilbert(t *testing.T) {
	doHilbert(t, 10)
}


func BenchmarkHilbert(b *testing.B) {
	for i := 0; i < b.N; i++ {
		doHilbert(nil, 10)
	}
}
