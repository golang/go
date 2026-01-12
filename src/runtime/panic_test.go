// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"slices"
	"strings"
	"testing"
	"time"
)

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
