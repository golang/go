// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is used to generate an object file which
// serves as test file for gcimporter_test.go.

package generics

type Any any

var x any

type T[A, B any] struct {
	Left  A
	Right B
}

var X T[int, string] = T[int, string]{1, "hi"}

func ToInt[P interface{ ~int }](p P) int { return int(p) }

var IntID = ToInt[int]

type G[C comparable] int

func ImplicitFunc[T ~int]() {}

type ImplicitType[T ~int] int
