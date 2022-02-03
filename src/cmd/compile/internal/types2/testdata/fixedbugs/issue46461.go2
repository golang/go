// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// test case 1
type T /* ERROR illegal cycle */ [U interface{ M() T[U] }] int

type X int

func (X) M() T[X] { return 0 }

// test case 2
type A /* ERROR illegal cycle */ [T interface{ A[T] }] interface{}

// test case 3
type A2 /* ERROR illegal cycle */ [U interface{ A2[U] }] interface{ M() A2[U] }

type I interface{ A2[I]; M() A2[I] }
