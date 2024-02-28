// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A[P any] [1]P

type B[P any] A /* ERROR "invalid recursive type" */ [P]

type C B[C]

// test case from issue

type Foo[T any] struct {
	baz T
}

type Bar[T any] struct {
	foo Foo /* ERROR "invalid recursive type" */ [T]
}

type Baz struct {
	bar Bar[Baz]
}
