// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
)

type A[T any] struct {
	B[int]
}

type B[T any] struct {
}

func (b B[T]) Bat() {
	t := new(T)
	if tt := reflect.TypeOf(t); tt.Kind() != reflect.Pointer || tt.Elem().Kind() != reflect.Int {
		panic("unexpected type, want: *int, got: "+tt.String())
	}
}

type Foo struct {
	A[string]
}
func main() {
	Foo{}.A.Bat()
	Foo{}.A.B.Bat()
	Foo{}.Bat()
}
