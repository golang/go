// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"os"
	_ "unsafe"
)

var modeflag = flag.String("mode", "", "mode to run in")

func main() {
	flag.Parse()
	switch *modeflag {
	case "simple":
		testSimple()
	case "goodexit":
		testGoodExit()
	case "badexit":
		testBadExit()
	case "panics":
		testPanics()
	case "callsexit":
		testHookCallsExit()
	default:
		panic("unknown mode")
	}
}

//go:linkname runtime_addExitHook runtime.addExitHook
func runtime_addExitHook(f func(), runOnNonZeroExit bool)

func testSimple() {
	f1 := func() { println("foo") }
	f2 := func() { println("bar") }
	runtime_addExitHook(f1, false)
	runtime_addExitHook(f2, false)
	// no explicit call to os.Exit
}

func testGoodExit() {
	f1 := func() { println("apple") }
	f2 := func() { println("orange") }
	runtime_addExitHook(f1, false)
	runtime_addExitHook(f2, false)
	// explicit call to os.Exit
	os.Exit(0)
}

func testBadExit() {
	f1 := func() { println("blog") }
	f2 := func() { println("blix") }
	f3 := func() { println("blek") }
	f4 := func() { println("blub") }
	f5 := func() { println("blat") }
	runtime_addExitHook(f1, false)
	runtime_addExitHook(f2, true)
	runtime_addExitHook(f3, false)
	runtime_addExitHook(f4, true)
	runtime_addExitHook(f5, false)
	os.Exit(1)
}

func testPanics() {
	f1 := func() { println("ok") }
	f2 := func() { panic("BADBADBAD") }
	f3 := func() { println("good") }
	runtime_addExitHook(f1, true)
	runtime_addExitHook(f2, true)
	runtime_addExitHook(f3, true)
	os.Exit(0)
}

func testHookCallsExit() {
	f1 := func() { println("ok") }
	f2 := func() { os.Exit(1) }
	f3 := func() { println("good") }
	runtime_addExitHook(f1, true)
	runtime_addExitHook(f2, true)
	runtime_addExitHook(f3, true)
	os.Exit(1)
}
