// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

// This file tests facts produced by ctrlflow.

var cond bool

var funcs = []func(){func() {}}

func a[A any]() { // want a:"noReturn"
	if cond {
		funcs[0]()
		b[A]()
	} else {
		for {
		}
	}
}

func b[B any]() { // want b:"noReturn"
	select {}
}

func c[A, B any]() { // want c:"noReturn"
	if cond {
		a[A]()
	} else {
		d[A, B]()
	}
}

func d[A, B any]() { // want d:"noReturn"
	b[B]()
}

type I[T any] interface {
	Id(T) T
}

func e[T any](i I[T], t T) T {
	return i.Id(t)
}

func k[T any](i I[T], t T) T { // want k:"noReturn"
	b[T]()
	return i.Id(t)
}

type T[X any] int

func (T[X]) method1() { // want method1:"noReturn"
	a[X]()
}

func (T[X]) method2() { // (may return)
	if cond {
		a[X]()
	} else {
		funcs[0]()
	}
}
