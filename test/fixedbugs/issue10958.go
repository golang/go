// +build !nacl
// buildrun -t 2  -gcflags=-d=ssa/insert_resched_checks/on,ssa/check/on

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This checks to see that call-free infinite loops do not
// block garbage collection.

package main

import (
	"runtime"
)

var someglobal1 int
var someglobal2 int
var someglobal3 int

//go:noinline
func f() {}

func standinacorner1() {
	for someglobal1&1 == 0 {
		someglobal1++
		someglobal1++
	}
}

func standinacorner2(i int) {
	// contains an irreducible loop containing changes to memory
	if i != 0 {
		goto midloop
	}

loop:
	if someglobal2&1 != 0 {
		goto done
	}
	someglobal2++
midloop:
	someglobal2++
	goto loop

done:
	return
}

func standinacorner3() {
	for someglobal3&1 == 0 {
		if someglobal3&2 != 0 {
			for someglobal3&3 == 2 {
				someglobal3++
				someglobal3++
				someglobal3++
				someglobal3++
			}
		}
		someglobal3++
		someglobal3++
		someglobal3++
		someglobal3++
	}
}

func main() {
	go standinacorner1()
	go standinacorner2(0)
	go standinacorner3()
	// println("About to stand in a corner1")
	for someglobal1 == 0 {
		runtime.Gosched()
	}
	// println("About to stand in a corner2")
	for someglobal2 == 0 {
		runtime.Gosched()
	}
	// println("About to stand in a corner3")
	for someglobal3 == 0 {
		runtime.Gosched()
	}
	// println("About to GC")
	runtime.GC()
	// println("Success")
}
