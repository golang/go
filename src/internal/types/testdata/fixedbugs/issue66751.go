// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S struct{}

func (*S) m(int) {}

func f[A interface {
	~*B
	m(C)
}, B, C any]() {
}

var _ = f[*S] // must be able to infer all remaining type arguments

// original test case from issue

type ptrTo[A any] interface{ ~*A }
type hasFoo[A any] interface{ foo(A) }
type both[A, B any] interface {
	ptrTo[A]
	hasFoo[B]
}

type fooer[A any] struct{}

func (f *fooer[A]) foo(A) {}

func withPtr[A ptrTo[B], B any]()       {}
func withFoo[A hasFoo[B], B any]()      {}
func withBoth[A both[B, C], B, C any]() {}

func _() {
	withPtr[*fooer[int]]()  // ok
	withFoo[*fooer[int]]()  // ok
	withBoth[*fooer[int]]() // should be able to infer C
}

// related test case reported in issue

type X struct{}

func (x X) M() int { return 42 }

func CallM1[T interface{ M() R }, R any](t T) R {
	return t.M()
}

func CallM2[T interface {
	X
	M() R
}, R any](t T) R {
	return t.M()
}

func _() {
	CallM1(X{}) // ok
	CallM2(X{}) // should be able to infer R
}
