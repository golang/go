// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// sanity check
type T[P any] struct {
	_ P
}

type S /* ERROR illegal cycle */ struct {
	_ T[S]
}

// simplified test
var _ B[A]

type A struct {
	_ B[string]
}

type B[P any] struct {
	_ C[P]
}

type C[P any] struct {
	_ P
}

// test case from issue
type T23 interface {
	~struct {
		Field0 T13[T15]
	}
}

type T1[P1 interface {
}] struct {
	Field2 P1
}

type T13[P2 interface {
}] struct {
	Field2 T1[P2]
}

type T15 struct {
	Field0 T13[string]
}
