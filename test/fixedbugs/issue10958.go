// +build !nacl,!js,disabled_see_issue_18589
// buildrun -t 10  -gcflags=-d=ssa/insert_resched_checks/on,ssa/check/on

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test is disabled because it flakes when run in all.bash
// on some platforms, but is useful standalone to verify
// that rescheduling checks are working (and we may wish
// to investigate the flake, since it suggests that the
// loop rescheduling check may not work right on those
// platforms).

// This checks to see that call-free infinite loops do not
// block garbage collection.  IF YOU RUN IT STANDALONE without
// -gcflags=-d=ssa/insert_resched_checks/on in a not-experimental
// build, it should hang.

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
