// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

//go:noinline
func f(x float64) float64 {
	return x
}

func inlineFma(x, y, z float64) float64 {
	return x + y*z
}

func main() {
	w, x, y := 1.0, 1.0, 1.0
	x = f(x + x/(1<<52))
	w = f(w / (1 << 27))
	y = f(y + y/(1<<52))
	w0 := f(2 * w * (1 - w))
	w1 := f(w * (1 + w))
	x = x + w0*w1
	x = inlineFma(x, w0, w1)
	y = y + f(w0*w1)
	y = y + f(w0*w1)
	fmt.Println(x, y, x-y)

	if x != y {
		os.Exit(1)
	}
}
