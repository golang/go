// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

type A struct{}

func (a A) Foo() {}

type B struct{}

func (b B) Foo() {}

func Do(a A, b B) map[I]I {
	m := make(map[I]I)
	m[a] = B{}
	m[b] = b
	return m
}

func Baz(a A, b B) {
	var x []I
	for k, v := range Do(a, b) {
		k.Foo()
		v.Foo()

		x = append(x, k)
	}

	x[len(x)-1].Foo()
}

// WANT:
// Baz: Do(a, b) -> Do; invoke t16.Foo() -> A.Foo, B.Foo; invoke t5.Foo() -> A.Foo, B.Foo; invoke t6.Foo() -> B.Foo
