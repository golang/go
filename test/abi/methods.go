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

type toobig struct {
	a, b, c string
}

//go:registerparams
//go:noinline
func (x *toobig) MagicMethodNameForTestingRegisterABI(y toobig, z toobig) toobig {
	return toobig{x.a, y.b, z.c}
}

type AnInterface interface {
	MagicMethodNameForTestingRegisterABI(y toobig, z toobig) toobig
}

//go:registerparams
//go:noinline
func I(a, b, c string) toobig {
	return toobig{a, b, c}
}

// AnIid prevents the compiler from figuring out what the interface really is.
//go:noinline
func AnIid(x AnInterface) AnInterface {
	return x
}

var tmp toobig

func main() {
	x := I("Ahoy", "1,", "2")
	y := I("3", "there,", "4")
	z := I("5", "6,", "Matey")
	tmp = x.MagicMethodNameForTestingRegisterABI(y, z)
	fmt.Println(tmp.a, tmp.b, tmp.c)
	tmp = AnIid(&x).MagicMethodNameForTestingRegisterABI(y, z)
	fmt.Println(tmp.a, tmp.b, tmp.c)
}
