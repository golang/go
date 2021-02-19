// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (!amd64 || appengine) && !gccgo
// +build !amd64 appengine
// +build !gccgo

package intsets

import "runtime"

// We compared three algorithms---Hacker's Delight, table lookup,
// and AMD64's SSE4.1 hardware POPCNT---on a 2.67GHz Xeon X5550.
//
// % GOARCH=amd64 go test -run=NONE -bench=Popcount
// POPCNT               5.12 ns/op
// Table                8.53 ns/op
// HackersDelight       9.96 ns/op
//
// % GOARCH=386 go test -run=NONE -bench=Popcount
// Table               10.4  ns/op
// HackersDelight       5.23 ns/op
//
// (AMD64's ABM1 hardware supports ntz and nlz too,
// but they aren't critical.)

// popcount returns the population count (number of set bits) of x.
func popcount(x word) int {
	if runtime.GOARCH == "386" {
		return popcountHD(uint32(x))
	}
	return popcountTable(x)
}
