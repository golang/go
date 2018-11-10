// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"strings"
	"testing"
)

func TestCaller(t *testing.T) {
	procs := runtime.GOMAXPROCS(-1)
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			for i := 0; i < 1000; i++ {
				testCallerFoo(t)
			}
			c <- true
		}()
		defer func() {
			<-c
		}()
	}
}

// These are marked noinline so that we can use FuncForPC
// in testCallerBar.
//go:noinline
func testCallerFoo(t *testing.T) {
	testCallerBar(t)
}

//go:noinline
func testCallerBar(t *testing.T) {
	for i := 0; i < 2; i++ {
		pc, file, line, ok := runtime.Caller(i)
		f := runtime.FuncForPC(pc)
		if !ok ||
			!strings.HasSuffix(file, "symtab_test.go") ||
			(i == 0 && !strings.HasSuffix(f.Name(), "testCallerBar")) ||
			(i == 1 && !strings.HasSuffix(f.Name(), "testCallerFoo")) ||
			line < 5 || line > 1000 ||
			f.Entry() >= pc {
			t.Errorf("incorrect symbol info %d: %t %d %d %s %s %d",
				i, ok, f.Entry(), pc, f.Name(), file, line)
		}
	}
}

func lineNumber() int {
	_, _, line, _ := runtime.Caller(1)
	return line // return 0 for error
}

// Do not add/remove lines in this block without updating the line numbers.
var firstLine = lineNumber() // 0
var (                        // 1
	lineVar1             = lineNumber()               // 2
	lineVar2a, lineVar2b = lineNumber(), lineNumber() // 3
)                        // 4
var compLit = []struct { // 5
	lineA, lineB int // 6
}{ // 7
	{ // 8
		lineNumber(), lineNumber(), // 9
	}, // 10
	{ // 11
		lineNumber(), // 12
		lineNumber(), // 13
	}, // 14
	{ // 15
		lineB: lineNumber(), // 16
		lineA: lineNumber(), // 17
	}, // 18
}                                     // 19
var arrayLit = [...]int{lineNumber(), // 20
	lineNumber(), lineNumber(), // 21
	lineNumber(), // 22
}                                  // 23
var sliceLit = []int{lineNumber(), // 24
	lineNumber(), lineNumber(), // 25
	lineNumber(), // 26
}                         // 27
var mapLit = map[int]int{ // 28
	29:           lineNumber(), // 29
	30:           lineNumber(), // 30
	lineNumber(): 31,           // 31
	lineNumber(): 32,           // 32
}                           // 33
var intLit = lineNumber() + // 34
	lineNumber() + // 35
	lineNumber() // 36
func trythis() { // 37
	recordLines(lineNumber(), // 38
		lineNumber(), // 39
		lineNumber()) // 40
}

// Modifications below this line are okay.

var l38, l39, l40 int

func recordLines(a, b, c int) {
	l38 = a
	l39 = b
	l40 = c
}

func TestLineNumber(t *testing.T) {
	trythis()
	for _, test := range []struct {
		name string
		val  int
		want int
	}{
		{"firstLine", firstLine, 0},
		{"lineVar1", lineVar1, 2},
		{"lineVar2a", lineVar2a, 3},
		{"lineVar2b", lineVar2b, 3},
		{"compLit[0].lineA", compLit[0].lineA, 9},
		{"compLit[0].lineB", compLit[0].lineB, 9},
		{"compLit[1].lineA", compLit[1].lineA, 12},
		{"compLit[1].lineB", compLit[1].lineB, 13},
		{"compLit[2].lineA", compLit[2].lineA, 17},
		{"compLit[2].lineB", compLit[2].lineB, 16},

		{"arrayLit[0]", arrayLit[0], 20},
		{"arrayLit[1]", arrayLit[1], 21},
		{"arrayLit[2]", arrayLit[2], 21},
		{"arrayLit[3]", arrayLit[3], 22},

		{"sliceLit[0]", sliceLit[0], 24},
		{"sliceLit[1]", sliceLit[1], 25},
		{"sliceLit[2]", sliceLit[2], 25},
		{"sliceLit[3]", sliceLit[3], 26},

		{"mapLit[29]", mapLit[29], 29},
		{"mapLit[30]", mapLit[30], 30},
		{"mapLit[31]", mapLit[31+firstLine] + firstLine, 31}, // nb it's the key not the value
		{"mapLit[32]", mapLit[32+firstLine] + firstLine, 32}, // nb it's the key not the value

		{"intLit", intLit - 2*firstLine, 34 + 35 + 36},

		{"l38", l38, 38},
		{"l39", l39, 39},
		{"l40", l40, 40},
	} {
		if got := test.val - firstLine; got != test.want {
			t.Errorf("%s on firstLine+%d want firstLine+%d (firstLine=%d, val=%d)",
				test.name, got, test.want, firstLine, test.val)
		}
	}
}

func TestNilName(t *testing.T) {
	defer func() {
		if ex := recover(); ex != nil {
			t.Fatalf("expected no nil panic, got=%v", ex)
		}
	}()
	if got := (*runtime.Func)(nil).Name(); got != "" {
		t.Errorf("Name() = %q, want %q", got, "")
	}
}
