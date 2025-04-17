// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// Y is the Y-combinator based on https://dreamsongs.com/Files/WhyOfY.pdf
func Y[Endo ~func(RecFct) RecFct, RecFct ~func(T) R, T, R any](f Endo) RecFct {

	type internal[RecFct ~func(T) R, T, R any] func(internal[RecFct, T, R]) RecFct

	g := func(h internal[RecFct, T, R]) RecFct {
		return func(t T) R {
			return f(h(h))(t)
		}
	}
	return g(g)
}

func main() {

	fct := Y(func(r func(int) int) func(int) int {
		return func(n int) int {
			if n <= 0 {
				return 1
			}
			return n * r(n-1)
		}
	})

	want := 3628800
	if got := fct(10); got != want {
		msg := fmt.Sprintf("unexpected result, got: %d, want: %d", got, want)
		panic(msg)
	}
}
