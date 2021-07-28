// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "a"

type R[T any] struct {
	F T
}

type S = R

type Sint = R[int]

type Simp = a.Rimp

type SimpString Simp[string]

func main() {
	var s S[int]
	if s.F != 0 {
		panic(s.F)
	}
	var s2 Sint
	if s2.F != 0 {
		panic(s2.F)
	}
	var s3 Simp[string]
	if s3.F != "" {
		panic(s3.F)
	}
	var s4 SimpString
	if s4.F != "" {
		panic(s4.F)
	}
}
