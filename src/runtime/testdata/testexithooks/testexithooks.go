// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"os"
	"internal/runtime/exithook"
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

func testSimple() {
	f1 := func() { println("foo") }
	f2 := func() { println("bar") }
	exithook.Add(exithook.Hook{F: f1})
	exithook.Add(exithook.Hook{F: f2})
	// no explicit call to os.Exit
}

func testGoodExit() {
	f1 := func() { println("apple") }
	f2 := func() { println("orange") }
	exithook.Add(exithook.Hook{F: f1})
	exithook.Add(exithook.Hook{F: f2})
	// explicit call to os.Exit
	os.Exit(0)
}

func testBadExit() {
	f1 := func() { println("blog") }
	f2 := func() { println("blix") }
	f3 := func() { println("blek") }
	f4 := func() { println("blub") }
	f5 := func() { println("blat") }
	exithook.Add(exithook.Hook{F: f1})
	exithook.Add(exithook.Hook{F: f2, RunOnFailure: true})
	exithook.Add(exithook.Hook{F: f3})
	exithook.Add(exithook.Hook{F: f4, RunOnFailure: true})
	exithook.Add(exithook.Hook{F: f5})
	os.Exit(1)
}

func testPanics() {
	f1 := func() { println("ok") }
	f2 := func() { panic("BADBADBAD") }
	f3 := func() { println("good") }
	exithook.Add(exithook.Hook{F: f1, RunOnFailure: true})
	exithook.Add(exithook.Hook{F: f2, RunOnFailure: true})
	exithook.Add(exithook.Hook{F: f3, RunOnFailure: true})
	os.Exit(0)
}

func testHookCallsExit() {
	f1 := func() { println("ok") }
	f2 := func() { os.Exit(1) }
	f3 := func() { println("good") }
	exithook.Add(exithook.Hook{F: f1, RunOnFailure: true})
	exithook.Add(exithook.Hook{F: f2, RunOnFailure: true})
	exithook.Add(exithook.Hook{F: f3, RunOnFailure: true})
	os.Exit(1)
}
