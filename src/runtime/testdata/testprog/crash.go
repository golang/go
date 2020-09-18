// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
)

func init() {
	register("Crash", Crash)
	register("DoublePanic", DoublePanic)
}

func test(name string) {
	defer func() {
		if x := recover(); x != nil {
			fmt.Printf(" recovered")
		}
		fmt.Printf(" done\n")
	}()
	fmt.Printf("%s:", name)
	var s *string
	_ = *s
	fmt.Print("SHOULD NOT BE HERE")
}

func testInNewThread(name string) {
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		test(name)
		c <- true
	}()
	<-c
}

func Crash() {
	runtime.LockOSThread()
	test("main")
	testInNewThread("new-thread")
	testInNewThread("second-new-thread")
	test("main-again")
}

type P string

func (p P) String() string {
	// Try to free the "YYY" string header when the "XXX"
	// panic is stringified.
	runtime.GC()
	runtime.GC()
	runtime.GC()
	return string(p)
}

// Test that panic message is not clobbered.
// See issue 30150.
func DoublePanic() {
	defer func() {
		panic(P("YYY"))
	}()
	panic(P("XXX"))
}
