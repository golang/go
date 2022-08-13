// run

//go:build !wasm
// +build !wasm

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

//go:registerparams
//go:noinline
func passStruct6(a Struct6) Struct6 {
	return a
}

type Struct6 struct {
	Struct1
}

type Struct1 struct {
	A, B, C uint
}

func main() {
	fmt.Println(passStruct6(Struct6{Struct1{1, 2, 3}}))
}
