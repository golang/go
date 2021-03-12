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
	type int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, uintptr, float32, float64
}

type MySlice []int

type _SliceOf[E any] interface {
	type []E
}

func _DoubleElems[S _SliceOf[E], E Number](s S) S {
	r := make(S, len(s))
	for i, v := range s {
		r[i] = v + v
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
}
