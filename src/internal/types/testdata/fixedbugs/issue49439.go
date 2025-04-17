// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "unsafe"

type T0 /* ERROR "invalid recursive type" */ [P T0[P]] struct{}

type T1 /* ERROR "invalid recursive type" */ [P T2[P]] struct{}
type T2[P T1[P]] struct{}

type T3 /* ERROR "invalid recursive type" */ [P interface{ ~struct{ f T3[int] } }] struct{}

// valid cycle in M
type N[P M[P]] struct{}
type M[Q any] struct { F *M[Q] }

// "crazy" case
type TC[P [unsafe.Sizeof(func() {
        type T [P [unsafe.Sizeof(func(){})]byte] struct{}
})]byte] struct{}

// test case from issue
type X /* ERROR "invalid recursive type" */ [T any, PT X[T]] interface{}
