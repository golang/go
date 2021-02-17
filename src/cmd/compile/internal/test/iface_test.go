// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import "testing"

// Test to make sure we make copies of the values we
// put in interfaces.

var x int

func TestEfaceConv1(t *testing.T) {
	a := 5
	i := interface{}(a)
	a += 2
	if got := i.(int); got != 5 {
		t.Errorf("wanted 5, got %d\n", got)
	}
}

func TestEfaceConv2(t *testing.T) {
	a := 5
	sink = &a
	i := interface{}(a)
	a += 2
	if got := i.(int); got != 5 {
		t.Errorf("wanted 5, got %d\n", got)
	}
}

func TestEfaceConv3(t *testing.T) {
	x = 5
	if got := e2int3(x); got != 5 {
		t.Errorf("wanted 5, got %d\n", got)
	}
}

//go:noinline
func e2int3(i interface{}) int {
	x = 7
	return i.(int)
}

func TestEfaceConv4(t *testing.T) {
	a := 5
	if got := e2int4(a, &a); got != 5 {
		t.Errorf("wanted 5, got %d\n", got)
	}
}

//go:noinline
func e2int4(i interface{}, p *int) int {
	*p = 7
	return i.(int)
}

type Int int

var y Int

type I interface {
	foo()
}

func (i Int) foo() {
}

func TestIfaceConv1(t *testing.T) {
	a := Int(5)
	i := interface{}(a)
	a += 2
	if got := i.(Int); got != 5 {
		t.Errorf("wanted 5, got %d\n", int(got))
	}
}

func TestIfaceConv2(t *testing.T) {
	a := Int(5)
	sink = &a
	i := interface{}(a)
	a += 2
	if got := i.(Int); got != 5 {
		t.Errorf("wanted 5, got %d\n", int(got))
	}
}

func TestIfaceConv3(t *testing.T) {
	y = 5
	if got := i2Int3(y); got != 5 {
		t.Errorf("wanted 5, got %d\n", int(got))
	}
}

//go:noinline
func i2Int3(i I) Int {
	y = 7
	return i.(Int)
}

func TestIfaceConv4(t *testing.T) {
	a := Int(5)
	if got := i2Int4(a, &a); got != 5 {
		t.Errorf("wanted 5, got %d\n", int(got))
	}
}

//go:noinline
func i2Int4(i I, p *Int) Int {
	*p = 7
	return i.(Int)
}

func BenchmarkEfaceInteger(b *testing.B) {
	sum := 0
	for i := 0; i < b.N; i++ {
		sum += i2int(i)
	}
	sink = sum
}

//go:noinline
func i2int(i interface{}) int {
	return i.(int)
}
