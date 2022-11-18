// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type I interface {
	Foo()
}

type A struct {
	I
}

func (a *A) Do() {
	a.Foo()
}

type B struct{}

func (b B) Foo() {}

func NewA(b B) *A {
	return &A{I: &b}
}

func Baz(b B) {
	a := NewA(b)
	a.Do()
}

// WANT:
// Baz: (*A).Do(t0) -> A.Do; NewA(b) -> NewA
// A.Do: invoke t1.Foo() -> B.Foo
