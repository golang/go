// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"reflect"
)

type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr | ~float32 | ~float64
}

type MySlice []int
type MyFloatSlice []float64

type _SliceOf[E any] interface {
	~[]E
}

func _DoubleElems[S _SliceOf[E], E Number](s S) S {
	r := make(S, len(s))
	for i, v := range s {
		r[i] = v + v
	}
	return r
}

// Test use of untyped constant in an expression with a generically-typed parameter
func _DoubleElems2[S _SliceOf[E], E Number](s S) S {
	r := make(S, len(s))
	for i, v := range s {
		r[i] = v * 2
	}
	return r
}

func main() {
	arg := MySlice{1, 2, 3}
	want := MySlice{2, 4, 6}
	got := _DoubleElems[MySlice, int](arg)
	if !reflect.DeepEqual(got, want) {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}

	// constraint type inference
	got = _DoubleElems[MySlice](arg)
	if !reflect.DeepEqual(got, want) {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}

	got = _DoubleElems(arg)
	if !reflect.DeepEqual(got, want) {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}

	farg := MyFloatSlice{1.2, 2.0, 3.5}
	fwant := MyFloatSlice{2.4, 4.0, 7.0}
	fgot := _DoubleElems(farg)
	if !reflect.DeepEqual(fgot, fwant) {
		panic(fmt.Sprintf("got %s, want %s", fgot, fwant))
	}

	fgot = _DoubleElems2(farg)
	if !reflect.DeepEqual(fgot, fwant) {
		panic(fmt.Sprintf("got %s, want %s", fgot, fwant))
	}
}
