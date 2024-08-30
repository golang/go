// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test break statements in a select.
// Gccgo had a bug in handling this.
// Test 1,2,3-case selects, so it covers both the general
// code path and the specialized optimizations for one-
// and two-case selects.

package main

var ch = make(chan int)

func main() {
	go func() {
		for {
			ch <- 5
		}
	}()

	select {
	case <-ch:
		break
		panic("unreachable")
	}

	select {
	default:
		break
		panic("unreachable")
	}

	select {
	case <-ch:
		break
		panic("unreachable")
	default:
		break
		panic("unreachable")
	}

	select {
	case <-ch:
		break
		panic("unreachable")
	case ch <- 10:
		panic("unreachable")
	default:
		break
		panic("unreachable")
	}
}
