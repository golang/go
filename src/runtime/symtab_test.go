// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"strings"
	"testing"
	"unsafe"
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

var dummy int

func inlined() {
	// Side effect to prevent elimination of this entire function.
	dummy = 42
}

// A function with an InlTree. Returns a PC within the function body.
//
// No inline to ensure this complete function appears in output.
//
//go:noinline
func tracebackFunc(t *testing.T) uintptr {
	// This body must be more complex than a single call to inlined to get
	// an inline tree.
	inlined()
	inlined()

	// Acquire a PC in this function.
	pc, _, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatalf("Caller(0) got ok false, want true")
	}

	return pc
}

// Test that CallersFrames handles PCs in the alignment region between
// functions (int 3 on amd64) without crashing.
//
// Go will never generate a stack trace containing such an address, as it is
// not a valid call site. However, the cgo traceback function passed to
// runtime.SetCgoTraceback may not be completely accurate and may incorrect
// provide PCs in Go code or the alignement region between functions.
//
// Go obviously doesn't easily expose the problematic PCs to running programs,
// so this test is a bit fragile. Some details:
//
// * tracebackFunc is our target function. We want to get a PC in the
//   alignment region following this function. This function also has other
//   functions inlined into it to ensure it has an InlTree (this was the source
//   of the bug in issue 44971).
//
// * We acquire a PC in tracebackFunc, walking forwards until FuncForPC says
//   we're in a new function. The last PC of the function according to FuncForPC
//   should be in the alignment region (assuming the function isn't already
//   perfectly aligned).
//
// This is a regression test for issue 44971.
func TestFunctionAlignmentTraceback(t *testing.T) {
	pc := tracebackFunc(t)

	// Double-check we got the right PC.
	f := runtime.FuncForPC(pc)
	if !strings.HasSuffix(f.Name(), "tracebackFunc") {
		t.Fatalf("Caller(0) = %+v, want tracebackFunc", f)
	}

	// Iterate forward until we find a different function. Back up one
	// instruction is (hopefully) an alignment instruction.
	for runtime.FuncForPC(pc) == f {
		pc++
	}
	pc--

	// Is this an alignment region filler instruction? We only check this
	// on amd64 for simplicity. If this function has no filler, then we may
	// get a false negative, but will never get a false positive.
	if runtime.GOARCH == "amd64" {
		code := *(*uint8)(unsafe.Pointer(pc))
		if code != 0xcc { // INT $3
			t.Errorf("PC %v code got %#x want 0xcc", pc, code)
		}
	}

	// Finally ensure that Frames.Next doesn't crash when processing this
	// PC.
	frames := runtime.CallersFrames([]uintptr{pc})
	frame, _ := frames.Next()
	if frame.Func != f {
		t.Errorf("frames.Next() got %+v want %+v", frame.Func, f)
	}
}
