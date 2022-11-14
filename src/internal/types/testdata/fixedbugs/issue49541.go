// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S[A, B any] struct {
	f int
}

func (S[A, B]) m() {}

// TODO(gri): with type-type inference enabled we should only report one error
// below. See issue #50588.

func _[A any](s S /* ERROR got 1 arguments but 2 type parameters */ [A]) {
	// we should see no follow-on errors below
	s.f = 1
	s.m()
}

// another test case from the issue

func _() {
	X(Interface[*F /* ERROR got 1 arguments but 2 type parameters */ [string]](Impl{}))
}

func X[Q Qer](fs Interface[Q]) {
}

type Impl struct{}

func (Impl) M() {}

type Interface[Q Qer] interface {
	M()
}

type Qer interface {
	Q()
}

type F[A, B any] struct{}

func (f *F[A, B]) Q() {}
