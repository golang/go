// run -gcflags=-d=maymorestack=main.mayMoreStack

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the maymorestack testing hook by injecting a hook that counts
// how many times it is called and checking that count.

package main

import "runtime"

var count uint32

//go:nosplit
func mayMoreStack() {
	count++
}

func main() {
	const wantCount = 128

	anotherFunc(wantCount - 1) // -1 because the call to main already counted

	if count == 0 {
		panic("mayMoreStack not called")
	} else if count != wantCount {
		println(count, "!=", wantCount)
		panic("wrong number of calls to mayMoreStack")
	}
}

//go:noinline
func anotherFunc(n int) {
	// Trigger a stack growth on at least some calls to
	// anotherFunc to test that mayMoreStack is called outside the
	// morestack loop. It's also important that it is called
	// before (not after) morestack, but that's hard to test.
	var x [1 << 10]byte

	if n > 1 {
		anotherFunc(n - 1)
	}

	runtime.KeepAlive(x)
}
