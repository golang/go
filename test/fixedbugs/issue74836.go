// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	a [20]int
}

func f(x [4]int) {
	g(T{}, x)
}

func g(t T, x [4]int)
