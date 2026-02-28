// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"sync/atomic"
)

var cnt32 int32

//go:noinline
func test32(a, b []int) bool {
	// Try to generate flag value, issue atomic
	// adds and then re-use the flag value to see if
	// the atomic add has clobbered them.
	atomic.AddInt32(&cnt32, 1)
	if len(a) == len(b) {
		atomic.AddInt32(&cnt32, 2)
	}
	atomic.AddInt32(&cnt32, 4)
	if len(a) >= len(b) {
		atomic.AddInt32(&cnt32, 8)
	}
	if len(a) <= len(b) {
		atomic.AddInt32(&cnt32, 16)
	}
	return atomic.LoadInt32(&cnt32) == 31
}

var cnt64 int64

//go:noinline
func test64(a, b []int) bool {
	// Try to generate flag value, issue atomic
	// adds and then re-use the flag value to see if
	// the atomic add has clobbered them.
	atomic.AddInt64(&cnt64, 1)
	if len(a) == len(b) {
		atomic.AddInt64(&cnt64, 2)
	}
	atomic.AddInt64(&cnt64, 4)
	if len(a) >= len(b) {
		atomic.AddInt64(&cnt64, 8)
	}
	if len(a) <= len(b) {
		atomic.AddInt64(&cnt64, 16)
	}
	return atomic.LoadInt64(&cnt64) == 31
}

func main() {
	if !test32([]int{}, []int{}) {
		panic("test32")
	}
	if !test64([]int{}, []int{}) {
		panic("test64")
	}
}
