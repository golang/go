// run -gcflags=-l=4

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
)

var skip int
var npcs int
var pcs = make([]uintptr, 32)

func f() {
	g()
}

func g() {
	h()
}

func h() {
	npcs = runtime.Callers(skip, pcs)
}

func testCallers(skp int) (frames []string) {
	skip = skp
	f()
	for i := 0; i < npcs; i++ {
		fn := runtime.FuncForPC(pcs[i] - 1)
		frames = append(frames, fn.Name())
		if fn.Name() == "main.main" {
			break
		}
	}
	return
}

func testCallersFrames(skp int) (frames []string) {
	skip = skp
	f()
	callers := pcs[:npcs]
	ci := runtime.CallersFrames(callers)
	for {
		frame, more := ci.Next()
		frames = append(frames, frame.Function)
		if !more || frame.Function == "main.main" {
			break
		}
	}
	return
}

var expectedFrames [][]string = [][]string{
	0: {"runtime.Callers", "main.h", "main.g", "main.f", "main.testCallers", "main.main"},
	1: {"main.h", "main.g", "main.f", "main.testCallers", "main.main"},
	2: {"main.g", "main.f", "main.testCallers", "main.main"},
	3: {"main.f", "main.testCallers", "main.main"},
	4: {"main.testCallers", "main.main"},
	5: {"main.main"},
}

var allFrames = []string{"runtime.Callers", "main.h", "main.g", "main.f", "main.testCallersFrames", "main.main"}

func same(xs, ys []string) bool {
	if len(xs) != len(ys) {
		return false
	}
	for i := range xs {
		if xs[i] != ys[i] {
			return false
		}
	}
	return true
}

func main() {
	for i := 0; i <= 5; i++ {
		frames := testCallers(i)
		expected := expectedFrames[i]
		if !same(frames, expected) {
			fmt.Printf("testCallers(%d):\n got %v\n want %v\n", i, frames, expected)
		}

		frames = testCallersFrames(i)
		expected = allFrames[i:]
		if !same(frames, expected) {
			fmt.Printf("testCallersFrames(%d):\n got %v\n want %v\n", i, frames, expected)
		}
	}
}
