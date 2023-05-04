// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// go:build ignore

package testdata

type F func()

func set[T [1]F | [2]F](arr *T, i int) {
	// Indexes into a pointer to an indexable type T and T does not have a coretype.
	// SSA instruction:	t0 = &arr[i]
	(*arr)[i] = bar
}

func bar() {
	print("here")
}

func Foo() {
	var arr [1]F
	set(&arr, 0)
	arr[0]()
}

// WANT:
// Foo: set[[1]testdata.F](t0, 0:int) -> set[[1]testdata.F]; t3() -> bar
