// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// correctness check: ensure that cycles through generic instantiations are detected
type T[P any] struct {
	_ P
}

type S /* ERROR "invalid recursive type" */ struct {
	_ T[S]
}

// simplified test 1

var _ A1[A1[string]]

type A1[P any] struct {
	_ B1[P]
}

type B1[P any] struct {
	_ P
}

// simplified test 2
var _ B2[A2]

type A2 struct {
	_ B2[string]
}

type B2[P any] struct {
	_ C2[P]
}

type C2[P any] struct {
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
