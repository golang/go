// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "fmt"

type Interface[T any] interface {
	m(Interface[T])
}

func f[S []Interface[T], T any](S) {}

func _() {
	var s []Interface[int]
	f(s) // panic here
}

// Larger example from issue

type InterfaceA[T comparable] interface {
	setData(string) InterfaceA[T]
}

type ImplA[T comparable] struct {
	data string
	args []any
}

func NewInterfaceA[T comparable](args ...any) InterfaceA[T] {
	return &ImplA[T]{
		data: fmt.Sprintf("%v", args...),
		args: args,
	}
}

func (k *ImplA[T]) setData(data string) InterfaceA[T] {
	k.data = data
	return k
}

func Foo[M ~map[InterfaceA[T]]V, T comparable, V any](m M) {
	// DO SOMETHING HERE
	return
}

func Bar() {
	keys := make([]InterfaceA[int], 0, 10)
	m := make(map[InterfaceA[int]]int)
	for i := 0; i < 10; i++ {
		keys = append(keys, NewInterfaceA[int](i))
		m[keys[i]] = i
	}

	Foo(m) // panic here
}
