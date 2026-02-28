// run

//go:build !wasm
// +build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test ensures that abi information producer and consumer agree about the
// order of registers for inputs.  T's registers should be I0, F0, I1, F1.

import "fmt"

type P struct {
	a int8
	x float64
}

type T struct {
	d, e P
}

//go:registerparams
//go:noinline
func G(t T) float64 {
	return float64(t.d.a+t.e.a) + t.d.x + t.e.x
}

func main() {
	x := G(T{P{10, 20}, P{30, 40}})
	if x != 100.0 {
		fmt.Printf("FAIL, Expected 100, got %f\n", x)
	}
}
