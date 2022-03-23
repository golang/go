// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./p"
	"fmt"
)

type I interface {
	Add(out *P)
}

type P struct {
	V *int32
}

type T struct{}

var x int32 = 42

func Int32x(i int32) *int32 {
	return &i
}

func (T) Add(out *P) {
	out.V = p.Int32(x) // inlined, p.i.2 moved to heap
}

var PP P
var out *P = &PP

func F(s I) interface{} {
	s.Add(out) // not inlined.
	return out
}

var s T

func main() {
	println("Starting")
	fmt.Sprint(new(int32))
	resp := F(s).(*P)
	println("Before, *resp.V=", *resp.V) // Trashes *resp.V in process of printing.
	println("After,  *resp.V=", *resp.V)
	if got, want := *resp.V, int32(42); got != want {
		fmt.Printf("FAIL, got %v, want %v", got, want)
	}
}
