// run

//go:build !wasm
// +build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// wasm is excluded because the compiler chatter about register abi pragma ends up
// on stdout, and causes the expected output to not match.

package main

import (
	"fmt"
)

var sink *string

type stringPair struct {
	a, b string
}

type stringPairPair struct {
	x, y stringPair
}

// The goal of this test is to be sure that the call arg/result expander works correctly
// for a corner case of passing a 2-nested struct that fits in registers to/from calls.

//go:registerparams
//go:noinline
func H(spp stringPairPair) string {
	return spp.x.a + " " + spp.x.b + " " + spp.y.a + " " + spp.y.b
}

//go:registerparams
//go:noinline
func G(a, b, c, d string) stringPairPair {
	return stringPairPair{stringPair{a, b}, stringPair{c, d}}
}

func main() {
	spp := G("this", "is", "a", "test")
	s := H(spp)
	gotVsWant(s, "this is a test")
}

func gotVsWant(got, want string) {
	if got != want {
		fmt.Printf("FAIL, got %s, wanted %s\n", got, want)
	}
}
