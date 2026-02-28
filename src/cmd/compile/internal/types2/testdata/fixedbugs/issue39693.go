// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type Number1 interface {
	// embedding non-interface types is permitted
	int
	float64
}

func Add1[T Number1](a, b T) T {
	return a /* ERROR not defined */ + b
}

type Number2 interface {
	int|float64
}

func Add2[T Number2](a, b T) T {
	return a + b
}
