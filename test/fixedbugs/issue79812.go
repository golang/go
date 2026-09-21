// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Foo interface {
	~float64
}

func bar[T Foo](x T, y func(a T) T) T {
	return y(0)
}

func f[T Foo](x T) T {
	return T(0 + 0i)
}

func g[T Foo](x T) T {
	return bar(0, f[T])
}

func main() {
	if got := g(0.0); got != 0 {
		println("unexpected result:", got)
		panic("FAILED")
	}
}
