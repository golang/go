// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"fmt"
	"internal/testenv"
	"strings"
	"testing"
)

//go:noinline
func rangeStringBytesCount(p []byte) int {
	n := 0
	for range string(p) {
		n++
	}
	return n
}

func TestRangeStringBytesAllocations(t *testing.T) {
	testenv.SkipIfOptimizationOff(t)
	for _, size := range []int{0, 1, 63, 64, 65, 256, 4096} {
		p := []byte(strings.Repeat("a", size))
		if n := testing.AllocsPerRun(10, func() {
			if got := rangeStringBytesCount(p); got != size {
				t.Fatalf("count = %d, want %d", got, size)
			}
		}); n != 0 {
			t.Errorf("size %d: %v allocations, want 0", size, n)
		}
	}
}

var rangeStringBytesSink int

func BenchmarkRangeStringBytes(b *testing.B) {
	for _, pattern := range []struct {
		name string
		text string
	}{
		{"ASCII", "a"},
		{"UTF8", "é世界"},
		{"Malformed", "\xff\xe2\x82a"},
	} {
		for _, size := range []int{64, 4096} {
			b.Run(fmt.Sprintf("%s/%d", pattern.name, size), func(b *testing.B) {
				p := []byte(strings.Repeat(pattern.text, (size+len(pattern.text)-1)/len(pattern.text))[:size])
				b.SetBytes(int64(len(p)))
				b.ReportAllocs()
				for b.Loop() {
					rangeStringBytesSink = rangeStringBytesCount(p)
				}
			})
		}
	}
}
