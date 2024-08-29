// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"maps"
	_ "unsafe"
)

func main() {
	for i := 0; i < 100; i++ {
		f()
	}
}

const NB = 4

func f() {
	// Make a map with NB buckets, at max capacity.
	// 6.5 entries/bucket.
	ne := NB * 13 / 2
	m := map[int]int{}
	for i := 0; i < ne; i++ {
		m[i] = i
	}

	// delete/insert a lot, to hopefully get lots of overflow buckets
	// and trigger a same-size grow.
	ssg := false
	for i := ne; i < ne+1000; i++ {
		delete(m, i-ne)
		m[i] = i
		if sameSizeGrow(m) {
			ssg = true
			break
		}
	}
	if !ssg {
		return
	}

	// Insert 1 more entry, which would ordinarily trigger a growth.
	// We can't grow while growing, so we instead go over our
	// target capacity.
	m[-1] = -1

	// Cloning in this state will make a map with a destination bucket
	// array twice the size of the source.
	_ = maps.Clone(m)
}

//go:linkname sameSizeGrow runtime.sameSizeGrowForIssue69110Test
func sameSizeGrow(m map[int]int) bool
