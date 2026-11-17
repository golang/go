// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package isimd is the leaf package (importing "simd") in the transitive-import
// test chain main1 -> relay -> isimd -> simd. It exports a constructor F
// returning an unexported struct that embeds simd.Int8s, plus non-SIMD
// variables derived from SIMD-dependent package-variable initialization chains.
package isimd

import "simd"

type t struct {
	simd.Int8s
}

//go:noinline
func id(i int32) int32 {
	return i
}

var a = id(4)                   // not constant, must evaluate before b
var b = simd.BroadcastInt32s(a) // evaluation depends on simd hardware
var c = b.ReduceSum()           // derive a non-simd value from a simd value
var d = c + 1                   // does this no-simd-dependence evaluation occur in the right order?

var i = simd.Int8s{}.Len()
var j = i // non-SIMD var chained off SIMD-initialized i

const three = 3

func F() t {
	return t{simd.BroadcastInt8s(three)}
}

// G returns the result of F().ReduceSum()
func G() int {
	return three * i
}

// H returns 4*(Int32s{}.Len())+1 == 17, 33, 65
func H() int32 {
	return d
}
