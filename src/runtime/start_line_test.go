// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"internal/testenv"
	"runtime"
	"testing"
)

// The tests in this file test the function start line metadata included in
// _func and inlinedCall. TestStartLine hard-codes the start lines of functions
// in this file. If code moves, the test will need to be updated.
//
// The "start line" of a function should be the line containing the func
// keyword.

func normalFunc() int {
	return callerStartLine(false)
}

func multilineDeclarationFunc() int {
	return multilineDeclarationFunc1(0, 0, 0)
}

//go:noinline
func multilineDeclarationFunc1(
	a, b, c int) int {
	return callerStartLine(false)
}

func blankLinesFunc() int {

	// Some
	// lines
	// without
	// code

	return callerStartLine(false)
}

func inlineFunc() int {
	return inlineFunc1()
}

func inlineFunc1() int {
	return callerStartLine(true)
}

var closureFn func() int

func normalClosure() int {
	// Assign to global to ensure this isn't inlined.
	closureFn = func() int {
		return callerStartLine(false)
	}
	return closureFn()
}

func inlineClosure() int {
	return func() int {
		return callerStartLine(true)
	}()
}

func TestStartLine(t *testing.T) {
	// We test inlined vs non-inlined variants. We can't do that if
	// optimizations are disabled.
	testenv.SkipIfOptimizationOff(t)

	testCases := []struct {
		name string
		fn   func() int
		want int
	}{
		{
			name: "normal",
			fn:   normalFunc,
			want: 21,
		},
		{
			name: "multiline-declaration",
			fn:   multilineDeclarationFunc,
			want: 30,
		},
		{
			name: "blank-lines",
			fn:   blankLinesFunc,
			want: 35,
		},
		{
			name: "inline",
			fn:   inlineFunc,
			want: 49,
		},
		{
			name: "normal-closure",
			fn:   normalClosure,
			want: 57,
		},
		{
			name: "inline-closure",
			fn:   inlineClosure,
			want: 64,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.fn()
			if got != tc.want {
				t.Errorf("start line got %d want %d", got, tc.want)
			}
		})
	}
}

//go:noinline
func callerStartLine(wantInlined bool) int {
	var pcs [1]uintptr
	n := runtime.Callers(2, pcs[:])
	if n != 1 {
		panic(fmt.Sprintf("no caller of callerStartLine? n = %d", n))
	}

	frames := runtime.CallersFrames(pcs[:])
	frame, _ := frames.Next()

	inlined := frame.Func == nil // Func always set to nil for inlined frames
	if wantInlined != inlined {
		panic(fmt.Sprintf("caller %s inlined got %v want %v", frame.Function, inlined, wantInlined))
	}

	return runtime.FrameStartLine(&frame)
}
