// run

//go:build !wasm
// +build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

var sink *string

type toobig struct {
	// 6 words will not SSA but will fit in registers
	a, b, c string
}

//go:registerparams
//go:noinline
func H(x toobig) string {
	return x.a + " " + x.b + " " + x.c
}

//go:registerparams
//go:noinline
func I(a, b, c string) toobig {
	return toobig{a, b, c}
}

func main() {
	s := H(toobig{"Hello", "there,", "World"})
	gotVsWant(s, "Hello there, World")
	fmt.Println(s)
	t := H(I("Ahoy", "there,", "Matey"))
	gotVsWant(t, "Ahoy there, Matey")
	fmt.Println(t)
}

func gotVsWant(got, want string) {
	if got != want {
		fmt.Printf("FAIL, got %s, wanted %s\n", got, want)
	}
}
