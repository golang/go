// errorcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code was incorrectly accepted by gccgo.

package main

type N string
type M string

const B N = "B"
const C M = "C"

func main() {
	q := B + C // ERROR "mismatched types|incompatible types"
	println(q)
}
