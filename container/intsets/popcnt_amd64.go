// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!appengine,!gccgo

package intsets

func popcnt(x word) int
func havePOPCNT() bool

var hasPOPCNT = havePOPCNT()

// popcount returns the population count (number of set bits) of x.
func popcount(x word) int {
	if hasPOPCNT {
		return popcnt(x)
	}
	return popcountTable(x) // faster than Hacker's Delight
}
