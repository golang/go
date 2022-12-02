// run
//go:build cgo

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "runtime/cgo"

type A struct {
	B
	_ cgo.Incomplete
}
type B struct{ x byte }
type I interface{ M() *B }

func (p *B) M() *B { return p }

var (
	a A
	i I = &a
)

func main() {
	got, want := i.M(), &a.B
	if got != want {
		println(got, "!=", want)
		panic("FAIL")
	}
}
