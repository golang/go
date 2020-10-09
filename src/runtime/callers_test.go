// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func f1(pan bool) []uintptr {
	return f2(pan) // line 15
}

func f2(pan bool) []uintptr {
	return f3(pan) // line 19
}

func f3(pan bool) []uintptr {
	if pan {
		panic("f3") // line 24
	}
	ret := make([]uintptr, 20)
	return ret[:runtime.Callers(0, ret)] // line 27
}

func testCallers(t *testing.T, pcs []uintptr, pan bool) {
	m := make(map[string]int, len(pcs))
	frames := runtime.CallersFrames(pcs)
	for {
		frame, more := frames.Next()
		if frame.Function != "" {
			m[frame.Function] = frame.Line
		}
		if !more {
			break
		}
	}

	var seen []string
	for k := range m {
		seen = append(seen, k)
	}
	t.Logf("functions seen: %s", strings.Join(seen, " "))

	var f3Line int
	if pan {
		f3Line = 24
	} else {
		f3Line = 27
	}
	want := []struct {
		name string
		line int
	}{
		{"f1", 15},
		{"f2", 19},
		{"f3", f3Line},
	}
	for _, w := range want {
		if got := m["runtime_test."+w.name]; got != w.line {
			t.Errorf("%s is line %d, want %d", w.name, got, w.line)
		}
	}
}

func testCallersEqual(t *testing.T, pcs []uintptr, want []string) {
	t.Helper()

	got := make([]string, 0, len(want))

	frames := runtime.CallersFrames(pcs)
	for {
		frame, more := frames.Next()
		if !more || len(got) >= len(want) {
			break
		}
		got = append(got, frame.Function)
	}
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("wanted %v, got %v", want, got)
	}
}

func TestCallers(t *testing.T) {
	testCallers(t, f1(false), false)
}

func TestCallersPanic(t *testing.T) {
	// Make sure we don't have any extra frames on the stack (due to
	// open-coded defer processing)
	want := []string{"runtime.Callers", "runtime_test.TestCallersPanic.func1",
		"runtime.gopanic", "runtime_test.f3", "runtime_test.f2", "runtime_test.f1",
		"runtime_test.TestCallersPanic"}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("did not panic")
		}
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallers(t, pcs, true)
		testCallersEqual(t, pcs, want)
	}()
	f1(true)
}

func TestCallersDoublePanic(t *testing.T) {
	// Make sure we don't have any extra frames on the stack (due to
	// open-coded defer processing)
	want := []string{"runtime.Callers", "runtime_test.TestCallersDoublePanic.func1.1",
		"runtime.gopanic", "runtime_test.TestCallersDoublePanic.func1", "runtime.gopanic", "runtime_test.TestCallersDoublePanic"}

	defer func() {
		defer func() {
			pcs := make([]uintptr, 20)
			pcs = pcs[:runtime.Callers(0, pcs)]
			if recover() == nil {
				t.Fatal("did not panic")
			}
			testCallersEqual(t, pcs, want)
		}()
		if recover() == nil {
			t.Fatal("did not panic")
		}
		panic(2)
	}()
	panic(1)
}

// Test that a defer after a successful recovery looks like it is called directly
// from the function with the defers.
func TestCallersAfterRecovery(t *testing.T) {
	want := []string{"runtime.Callers", "runtime_test.TestCallersAfterRecovery.func1", "runtime_test.TestCallersAfterRecovery"}

	defer func() {
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallersEqual(t, pcs, want)
	}()
	defer func() {
		if recover() == nil {
			t.Fatal("did not recover from panic")
		}
	}()
	panic(1)
}

func TestCallersAbortedPanic(t *testing.T) {
	want := []string{"runtime.Callers", "runtime_test.TestCallersAbortedPanic.func2", "runtime_test.TestCallersAbortedPanic"}

	defer func() {
		r := recover()
		if r != nil {
			t.Fatalf("should be no panic remaining to recover")
		}
	}()

	defer func() {
		// panic2 was aborted/replaced by panic1, so when panic2 was
		// recovered, there is no remaining panic on the stack.
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallersEqual(t, pcs, want)
	}()
	defer func() {
		r := recover()
		if r != "panic2" {
			t.Fatalf("got %v, wanted %v", r, "panic2")
		}
	}()
	defer func() {
		// panic2 aborts/replaces panic1, because it is a recursive panic
		// that is not recovered within the defer function called by
		// panic1 panicking sequence
		panic("panic2")
	}()
	panic("panic1")
}

func TestCallersAbortedPanic2(t *testing.T) {
	want := []string{"runtime.Callers", "runtime_test.TestCallersAbortedPanic2.func2", "runtime_test.TestCallersAbortedPanic2"}
	defer func() {
		r := recover()
		if r != nil {
			t.Fatalf("should be no panic remaining to recover")
		}
	}()
	defer func() {
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallersEqual(t, pcs, want)
	}()
	func() {
		defer func() {
			r := recover()
			if r != "panic2" {
				t.Fatalf("got %v, wanted %v", r, "panic2")
			}
		}()
		func() {
			defer func() {
				// Again, panic2 aborts/replaces panic1
				panic("panic2")
			}()
			panic("panic1")
		}()
	}()
}

func TestCallersNilPointerPanic(t *testing.T) {
	// Make sure we don't have any extra frames on the stack (due to
	// open-coded defer processing)
	want := []string{"runtime.Callers", "runtime_test.TestCallersNilPointerPanic.func1",
		"runtime.gopanic", "runtime.panicmem", "runtime.sigpanic",
		"runtime_test.TestCallersNilPointerPanic"}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("did not panic")
		}
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallersEqual(t, pcs, want)
	}()
	var p *int
	if *p == 3 {
		t.Fatal("did not see nil pointer panic")
	}
}

func TestCallersDivZeroPanic(t *testing.T) {
	// Make sure we don't have any extra frames on the stack (due to
	// open-coded defer processing)
	want := []string{"runtime.Callers", "runtime_test.TestCallersDivZeroPanic.func1",
		"runtime.gopanic", "runtime.panicdivide",
		"runtime_test.TestCallersDivZeroPanic"}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("did not panic")
		}
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallersEqual(t, pcs, want)
	}()
	var n int
	if 5/n == 1 {
		t.Fatal("did not see divide-by-sizer panic")
	}
}

func TestCallersDeferNilFuncPanic(t *testing.T) {
	// Make sure we don't have any extra frames on the stack. We cut off the check
	// at runtime.sigpanic, because non-open-coded defers (which may be used in
	// non-opt or race checker mode) include an extra 'deferreturn' frame (which is
	// where the nil pointer deref happens).
	state := 1
	want := []string{"runtime.Callers", "runtime_test.TestCallersDeferNilFuncPanic.func1",
		"runtime.gopanic", "runtime.panicmem", "runtime.sigpanic"}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("did not panic")
		}
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallersEqual(t, pcs, want)
		if state == 1 {
			t.Fatal("nil defer func panicked at defer time rather than function exit time")
		}

	}()
	var f func()
	defer f()
	// Use the value of 'state' to make sure nil defer func f causes panic at
	// function exit, rather than at the defer statement.
	state = 2
}

// Same test, but forcing non-open-coded defer by putting the defer in a loop.  See
// issue #36050
func TestCallersDeferNilFuncPanicWithLoop(t *testing.T) {
	state := 1
	want := []string{"runtime.Callers", "runtime_test.TestCallersDeferNilFuncPanicWithLoop.func1",
		"runtime.gopanic", "runtime.panicmem", "runtime.sigpanic", "runtime.deferreturn", "runtime_test.TestCallersDeferNilFuncPanicWithLoop"}

	defer func() {
		if r := recover(); r == nil {
			t.Fatal("did not panic")
		}
		pcs := make([]uintptr, 20)
		pcs = pcs[:runtime.Callers(0, pcs)]
		testCallersEqual(t, pcs, want)
		if state == 1 {
			t.Fatal("nil defer func panicked at defer time rather than function exit time")
		}

	}()

	for i := 0; i < 1; i++ {
		var f func()
		defer f()
	}
	// Use the value of 'state' to make sure nil defer func f causes panic at
	// function exit, rather than at the defer statement.
	state = 2
}
