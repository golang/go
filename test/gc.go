// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple test of the garbage collector.

package main

import "runtime"

func mk2() {
	b := new([10000]byte)
	_ = b
	//	println(b, "stored at", &b)
}

func mk1() { mk2() }

func main() {
	for i := 0; i < 10; i++ {
		mk1()
		runtime.GC()
	}
}
