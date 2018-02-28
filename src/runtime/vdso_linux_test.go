// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux
// +build 386 amd64

package runtime_test

import (
	"testing"
	"time"
	_ "unsafe"
)

// These tests are a little risky because they overwrite the __vdso_clock_gettime_sym value.
// It's normally initialized at startup and remains unchanged after that.

//go:linkname __vdso_clock_gettime_sym runtime.__vdso_clock_gettime_sym
var __vdso_clock_gettime_sym uintptr

func TestClockVDSOAndFallbackPaths(t *testing.T) {
	// Check that we can call walltime() and nanotime() with and without their (1st) fast-paths.
	// This just checks that fast and fallback paths can be called, rather than testing their
	// results.
	//
	// Call them indirectly via time.Now(), so we don't need auxiliary .s files to allow us to
	// use go:linkname to refer to the functions directly.

	save := __vdso_clock_gettime_sym
	if save == 0 {
		t.Log("__vdso_clock_gettime symbol not found; fallback path will be used by default")
	}

	// Call with fast-path enabled (if vDSO symbol found at startup)
	time.Now()

	// Call with fast-path disabled
	__vdso_clock_gettime_sym = 0
	time.Now()
	__vdso_clock_gettime_sym = save
}

func BenchmarkClockVDSOAndFallbackPaths(b *testing.B) {
	run := func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			// Call via time.Now() - see comment in test above.
			time.Now()
		}
	}

	save := __vdso_clock_gettime_sym
	b.Run("vDSO", run)
	__vdso_clock_gettime_sym = 0
	b.Run("Fallback", run)
	__vdso_clock_gettime_sym = save
}

func BenchmarkTimeNow(b *testing.B) {
	for i := 0; i < b.N; i++ {
		time.Now()
	}
}
