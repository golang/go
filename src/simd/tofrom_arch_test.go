// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64 || wasm)

package simd_test

import (
	"fmt"
	"internal/testenv"
	"simd"
	"simd/archsimd"
	"strings"
	"testing"
)

// These platforms all support Int32x4 and Mask32x4.

func roundTripInt32s(x simd.Int32s) simd.Int32s {
	if a, ok := x.ToArch().(archsimd.Int32x4); ok {
		return simd.Int32sFromArch(a)
	}
	return x
}

func roundTripMask32s(m simd.Mask32s) simd.Mask32s {
	if a, ok := m.ToArch().(archsimd.Mask32x4); ok {
		return simd.Mask32sFromArch(a)
	}
	return m
}

func TestToFromArch128(t *testing.T) {
	s := make([]int32, 16)
	for i := range s {
		s[i] = int32(i + 1)
	}
	x := simd.LoadInt32s(s)
	got := make([]int32, x.Len())
	roundTripInt32s(x).Store(got)
	for i := range got {
		if got[i] != s[i] {
			t.Fatalf("roundTripInt32s lane %d = %d, want %d", i, got[i], s[i])
		}
	}

	zero := simd.BroadcastInt32s(0)
	m := x.Greater(zero)
	if roundTripMask32s(m).String() != m.String() {
		t.Fatalf("roundTripMask32s mismatch: got %s, want %s", roundTripMask32s(m), m)
	}
}

func TestFromArchEmulatedPanic(t *testing.T) {
	if !simd.Emulated() {
		// Self-invoke in emulated mode.
		cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestFromArchEmulatedPanic$")
		cmd.Env = append(cmd.Environ(), "GODEBUG=simd=0")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("GODEBUG=simd=0 subprocess failed: %v\n%s", err, out)
		}
		return
	}

	callFromArch128 := func() simd.Int32s {
		return simd.Int32sFromArch(archsimd.Int32x4{})
	}
	expectFromArchPanic(t, func() { _ = callFromArch128() },
		"simd.Int32sFromArch: cannot convert from archsimd.Int32x4 when simd.Emulated() is true")
}

func expectFromArchPanic(t *testing.T, fn func(), want string) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic %q, did not panic", want)
		}
		got := fmt.Sprint(r)
		if strings.Contains(got, "bridge.") {
			t.Errorf("panic %q leaks internal bridge package", got)
		}
		if got != want {
			t.Errorf("panic = %q, want %q", got, want)
		}
	}()
	fn()
}
