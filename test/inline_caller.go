// run -gcflags -l=4

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
)

type frame struct {
	pc   uintptr
	file string
	line int
	ok   bool
}

var (
	skip        int
	globalFrame frame
)

func f() {
	g() // line 27
}

func g() {
	h() // line 31
}

func h() {
	x := &globalFrame
	x.pc, x.file, x.line, x.ok = runtime.Caller(skip) // line 36
}

//go:noinline
func testCaller(skp int) frame {
	skip = skp
	f() // line 42
	frame := globalFrame
	if !frame.ok {
		panic(fmt.Sprintf("skip=%d runtime.Caller failed", skp))
	}
	return frame
}

type wantFrame struct {
	funcName string
	line     int
}

// -1 means don't care
var expected = []wantFrame{
	0: {"main.testCaller", 36},
	1: {"main.testCaller", 31},
	2: {"main.testCaller", 27},
	3: {"main.testCaller", 42},
	4: {"main.main", 68},
	5: {"runtime.main", -1},
	6: {"runtime.goexit", -1},
}

func main() {
	for i := 0; i <= 6; i++ {
		frame := testCaller(i) // line 68
		fn := runtime.FuncForPC(frame.pc)
		if expected[i].line >= 0 && frame.line != expected[i].line {
			panic(fmt.Sprintf("skip=%d expected line %d, got line %d", i, expected[i].line, frame.line))
		}
		if fn.Name() != expected[i].funcName {
			panic(fmt.Sprintf("skip=%d expected function %s, got %s", i, expected[i].funcName, fn.Name()))
		}
	}
}
