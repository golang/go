// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"
)

// Check that 64-bit indices retain both words and their signedness when
// reporting bounds failures on 32-bit architectures.
func TestBoundsPanic64(t *testing.T) {
	s := make([]byte, 3, 5)
	a := [3]byte{}
	str := "abc"
	tests := []struct {
		name     string
		signed   func(int64)
		unsigned func(uint64)
		positive string
		negative string
	}{
		{"index", func(i int64) { _ = s[i] }, func(i uint64) { _ = s[i] },
			"index out of range [%d] with length 3", "index out of range [%d]"},
		{"sliceLen", func(i int64) { _ = str[:i] }, func(i uint64) { _ = str[:i] },
			"slice bounds out of range [:%d] with length 3", "slice bounds out of range [:%d]"},
		{"sliceCap", func(i int64) { _ = s[:i] }, func(i uint64) { _ = s[:i] },
			"slice bounds out of range [:%d] with capacity 5", "slice bounds out of range [:%d]"},
		{"sliceLow", func(i int64) { _ = s[i:2] }, func(i uint64) { _ = s[i:2] },
			"slice bounds out of range [%d:2]", "slice bounds out of range [%d:]"},
		{"slice3Len", func(i int64) { _ = a[:1:i] }, func(i uint64) { _ = a[:1:i] },
			"slice bounds out of range [::%d] with length 3", "slice bounds out of range [::%d]"},
		{"slice3Cap", func(i int64) { _ = s[:1:i] }, func(i uint64) { _ = s[:1:i] },
			"slice bounds out of range [::%d] with capacity 5", "slice bounds out of range [::%d]"},
		{"slice3High", func(i int64) { _ = s[:i:2] }, func(i uint64) { _ = s[:i:2] },
			"slice bounds out of range [:%d:2]", "slice bounds out of range [:%d:]"},
		{"slice3Low", func(i int64) { _ = s[i:1:2] }, func(i uint64) { _ = s[i:1:2] },
			"slice bounds out of range [%d:1:]", "slice bounds out of range [%d::]"},
	}
	check := func(t *testing.T, f func(), want string) {
		t.Helper()
		defer func() {
			r := recover()
			err, ok := r.(runtime.Error)
			if !ok {
				t.Fatalf("panic = %v (%T), want runtime.Error", r, r)
			}
			if got := err.Error(); got != "runtime error: "+want {
				t.Fatalf("panic = %q, want %q", got, "runtime error: "+want)
			}
		}()
		f()
	}
	for _, tt := range tests {
		for _, i := range []int64{1 << 32, 1<<32 + 3, -1<<32 + 3, -1 << 63} {
			t.Run(fmt.Sprintf("%s/int64/%d", tt.name, i), func(t *testing.T) {
				format := tt.positive
				if i < 0 {
					format = tt.negative
				}
				check(t, func() { tt.signed(i) }, fmt.Sprintf(format, i))
			})
		}
		for _, i := range []uint64{1 << 32, 1<<32 + 3, 1 << 63, ^uint64(0)} {
			t.Run(fmt.Sprintf("%s/uint64/%d", tt.name, i), func(t *testing.T) {
				check(t, func() { tt.unsigned(i) }, fmt.Sprintf(tt.positive, i))
			})
		}
	}
}

// Test that panics print out the underlying value
// when the underlying kind is directly printable.
// Issue: https://golang.org/issues/37531
func TestPanicWithDirectlyPrintableCustomTypes(t *testing.T) {
	tests := []struct {
		name            string
		wantPanicPrefix string
	}{
		{"panicCustomBool", `panic: main.MyBool(true)`},
		{"panicCustomComplex128", `panic: main.MyComplex128(32.1+10i)`},
		{"panicCustomComplex64", `panic: main.MyComplex64(0.11+3i)`},
		{"panicCustomFloat32", `panic: main.MyFloat32(-93.7)`},
		{"panicCustomFloat64", `panic: main.MyFloat64(-93.7)`},
		{"panicCustomInt", `panic: main.MyInt(93)`},
		{"panicCustomInt8", `panic: main.MyInt8(93)`},
		{"panicCustomInt16", `panic: main.MyInt16(93)`},
		{"panicCustomInt32", `panic: main.MyInt32(93)`},
		{"panicCustomInt64", `panic: main.MyInt64(93)`},
		{"panicCustomString", `panic: main.MyString("Panic` + "\n\t" + `line two")`},
		{"panicCustomUint", `panic: main.MyUint(93)`},
		{"panicCustomUint8", `panic: main.MyUint8(93)`},
		{"panicCustomUint16", `panic: main.MyUint16(93)`},
		{"panicCustomUint32", `panic: main.MyUint32(93)`},
		{"panicCustomUint64", `panic: main.MyUint64(93)`},
		{"panicCustomUintptr", `panic: main.MyUintptr(93)`},
		{"panicDeferFatal", "panic: runtime.errorString(\"invalid memory address or nil pointer dereference\")\n\tfatal error: sync: unlock of unlocked mutex"},
		{"panicDoublieDeferFatal", "panic: runtime.errorString(\"invalid memory address or nil pointer dereference\") [recovered, repanicked]\n\tfatal error: sync: unlock of unlocked mutex"},
	}

	for _, tt := range tests {
		t := t
		t.Run(tt.name, func(t *testing.T) {
			output := runTestProg(t, "testprog", tt.name)
			if !strings.HasPrefix(output, tt.wantPanicPrefix) {
				t.Fatalf("%q\nis not present in\n%s", tt.wantPanicPrefix, output)
			}
		})
	}
}

func TestPanicRecoverSpeed(t *testing.T) {
	// For issue 77062.
	t.Skip("This test is too flaky at the moment. But it does normally pass. Suggestions for making it less flaky are welcome.")

	// Recursive function that does defer/recover/repanic.
	var f func(int)
	f = func(n int) {
		if n == 0 {
			panic("done")
		}
		defer func() {
			err := recover()
			panic(err)
		}()
		f(n - 1)
	}

	time := func(f func()) time.Duration {
		var times []time.Duration
		for range 10 {
			start := time.Now()
			f()
			times = append(times, time.Since(start))
		}
		slices.Sort(times)
		times = times[1 : len(times)-1] // skip high and low, to reduce noise
		var avg time.Duration
		for _, v := range times {
			avg += v / time.Duration(len(times))
		}
		return avg
	}

	a := time(func() {
		defer func() { recover() }()
		f(1024)
	})
	b := time(func() {
		defer func() { recover() }()
		f(2048)
	})
	m := b.Seconds() / a.Seconds()
	t.Logf("a: %v, b: %v, m: %v", a, b, m)
	if m > 3.5 {
		t.Errorf("more than 2x time increase: %v", m)
	}
}

// Test that panics with nil arguments produce the prefix
// "runtime error:" per https://golang.org/issues/63813.
func TestPanicNilErrorPrefix(t *testing.T) {
	tests := []struct {
		name      string
		wantPanic string
		fn        func()
	}{
		{
			name:      "panic(nil)",
			wantPanic: "runtime error: panic with nil argument",
			fn: func() {
				panic(nil)
			},
		},
		{
			name:      "panic((any)(nil))",
			wantPanic: "runtime error: panic with nil argument",
			fn: func() {
				var foo any = nil
				panic(foo)
			},
		},
		{
			name:      "panic((error)(nil))",
			wantPanic: "runtime error: panic with nil argument",
			fn: func() {
				var err error
				panic(err)
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("expected a panic")
				}

				re, ok := r.(runtime.Error)
				if !ok {
					t.Fatalf("wrong panic type: got %T, want runtime.Error", r)
				}
				if !strings.Contains(re.Error(), "runtime error: panic called with nil argument") {
					t.Fatalf("mismatched message, missing `runtime error: panic with nil`, : got:\n%s", re.Error())
				}
			}()

			tt.fn()
		})
	}
}

// Issue 81959: nil-pointer panic in a promoted-method wrapper should not cause
// "traceback did not unwind completely" when an intervening frame has a linked defer.
type wrapperTestI interface {
	m() wrapperTestI
}

type wrapperTestT struct {
	x int
}

func (t *wrapperTestT) m() wrapperTestI {
	return t
}

type wrapperTestW struct {
	wrapperTestT
}

//go:noinline
func testWrapperPanicLinked(i wrapperTestI) wrapperTestI {
	for range 1 {
		// Use a loop defer so open-coded defers are disabled, forcing a linked defer.
		defer func() {}()
	}
	return i.m()
}

//go:noinline
func testWrapperPanicOpen(i wrapperTestI) wrapperTestI {
	defer func() {}()
	return i.m()
}

func TestWrapperPanicUnwind(t *testing.T) {
	t.Run("OpenCoded", func(t *testing.T) {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic, got nil")
			}
		}()
		testWrapperPanicOpen((*wrapperTestW)(nil))
	})
	t.Run("Linked", func(t *testing.T) {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("expected panic, got nil")
			}
		}()
		testWrapperPanicLinked((*wrapperTestW)(nil))
	})
}
