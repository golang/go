// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64

package maps_test

import (
	"fmt"
	"testing"
	"unsafe"

	"internal/runtime/maps"
)

var sink uintptr

// BenchmarkHashBakeoff measures the AES and scalar memory hashers at
// various sizes to try to empirically determine when one becomes better than
// the other, for some target uarch. Results are very uarch-dependent!
//
// The datapoints should be compared something like benchstat which uses the
// appropriate statistical tests to knock out outliers.
//
// Latency (i.e., serial pipeline performance) matters for probing, because
// there is a data dependency between the hash and the probe sequence. We can
// measure this by making each iteration of the benchmark depend on the previous
// one. This tends to favor scalar-only hashing more.
//
// Throughput (i.e., how long matters when many independent things are being
// hashed, resulting in better IPC. We measure this by using a seed of 0 for
// each iteration. This tends to favor AES more.
func BenchmarkHashBakeoff(b *testing.B) {
	if !maps.AeshashEnabled() {
		b.Skip("AES hashing not available on this machine")
	}

	buf := make([]byte, 1024+8)
	for i := range buf {
		buf[i] = byte(i * 63)
	}
	p := unsafe.Pointer(unsafe.SliceData(buf))

	var sizes = []uintptr{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 100, 104, 108, 112, 116,
		120, 124, 128, 192, 256, 512, 1024,
	}

	for _, s := range sizes {
		b.Run(fmt.Sprintf("scalar/latency/%d", s), func(b *testing.B) {
			var h uintptr
			for b.Loop() {
				h = maps.MemHashFallback(p, h, s)
			}
			sink = h
		})
	}
	for _, s := range sizes {
		b.Run(fmt.Sprintf("scalar/throughput/%d", s), func(b *testing.B) {
			var h uintptr
			for b.Loop() {
				h ^= maps.MemHashFallback(p, 0, s)
			}
			sink = h
		})
	}
	for _, s := range sizes {
		b.Run(fmt.Sprintf("aes/latency/%d", s), func(b *testing.B) {
			var h uintptr
			for b.Loop() {
				h = maps.MemHashAES(p, h, s)
			}
			sink = h
		})
	}
	for _, s := range sizes {
		b.Run(fmt.Sprintf("aes/throughput/%d", s), func(b *testing.B) {
			var h uintptr
			for i := 0; i < b.N; i++ {
				h ^= maps.MemHashAES(p, 0, s)
			}
			sink = h
		})
	}
}
