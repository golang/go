// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"testing"
)

var logosLoopStrideSink int

func BenchmarkLogosLoopStride(b *testing.B) {
	outer := 64
	inner := 4096
	before := uint64(testing.AllocsPerRun(1, func() {
		for i := 0; i < 1; i++ {
			runLoopStrideWindow(inner)
		}
	}))
	if before != 0 {
		b.Fatalf("stride window of %d allocated %d times, want allocation-free back edges", inner, before)
	}
	b.SetBytes(int64(outer * inner))
	b.ResetTimer()
	for b.Loop() {
		for range outer {
			runLoopStrideWindow(inner)
		}
	}
}

func runLoopStrideWindow(n int) {
	acc := logosLoopStrideSink
	for i := range n {
		acc = acc*3 + i
	}
	logosLoopStrideSink = acc
}
