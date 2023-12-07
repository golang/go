// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"strconv"
	"testing"
	_ "unsafe" // for go:linkname
)

func TestReadRandom(t *testing.T) {
	if *ReadRandomFailed {
		switch GOOS {
		default:
			t.Fatalf("readRandom failed at startup")
		case "plan9":
			// ok
		}
	}
}

func BenchmarkFastrand(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Fastrand()
		}
	})
}

func BenchmarkFastrand64(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Fastrand64()
		}
	})
}

func BenchmarkFastrandHashiter(b *testing.B) {
	var m = make(map[int]int, 10)
	for i := 0; i < 10; i++ {
		m[i] = i
	}
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			for range m {
				break
			}
		}
	})
}

var sink32 uint32

func BenchmarkFastrandn(b *testing.B) {
	for n := uint32(2); n <= 5; n++ {
		b.Run(strconv.Itoa(int(n)), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink32 = Fastrandn(n)
			}
		})
	}
}

//go:linkname fastrand runtime.fastrand
func fastrand() uint32

//go:linkname fastrandn runtime.fastrandn
func fastrandn(uint32) uint32

//go:linkname fastrand64 runtime.fastrand64
func fastrand64() uint64

func TestLegacyFastrand(t *testing.T) {
	// Testing mainly that the calls work at all,
	// but check that all three don't return the same number (1 in 2^64 chance)
	{
		x, y, z := fastrand(), fastrand(), fastrand()
		if x == y && y == z {
			t.Fatalf("fastrand three times = %#x, %#x, %#x, want different numbers", x, y, z)
		}
	}
	{
		x, y, z := fastrandn(1e9), fastrandn(1e9), fastrandn(1e9)
		if x == y && y == z {
			t.Fatalf("fastrandn three times = %#x, %#x, %#x, want different numbers", x, y, z)
		}
	}
	{
		x, y, z := fastrand64(), fastrand64(), fastrand64()
		if x == y && y == z {
			t.Fatalf("fastrand64 three times = %#x, %#x, %#x, want different numbers", x, y, z)
		}
	}
}
