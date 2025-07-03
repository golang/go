// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(61395): move these tests to atomic_test.go once And/Or have
// implementations for all architectures.
package atomic_test

import (
	"internal/runtime/atomic"
	"testing"
)

func TestAnd32(t *testing.T) {
	// Basic sanity check.
	x := uint32(0xffffffff)
	for i := uint32(0); i < 32; i++ {
		old := x
		v := atomic.And32(&x, ^(1 << i))
		if r := uint32(0xffffffff) << (i + 1); x != r || v != old {
			t.Fatalf("clearing bit %#x: want %#x, got new %#x and old %#v", uint32(1<<i), r, x, v)
		}
	}

	// Set every bit in array to 1.
	a := make([]uint32, 1<<12)
	for i := range a {
		a[i] = 0xffffffff
	}

	// Clear array bit-by-bit in different goroutines.
	done := make(chan bool)
	for i := 0; i < 32; i++ {
		m := ^uint32(1 << i)
		go func() {
			for i := range a {
				atomic.And(&a[i], m)
			}
			done <- true
		}()
	}
	for i := 0; i < 32; i++ {
		<-done
	}

	// Check that the array has been totally cleared.
	for i, v := range a {
		if v != 0 {
			t.Fatalf("a[%v] not cleared: want %#x, got %#x", i, uint32(0), v)
		}
	}
}

func TestAnd64(t *testing.T) {
	// Basic sanity check.
	x := uint64(0xffffffffffffffff)
	sink = &x
	for i := uint64(0); i < 64; i++ {
		old := x
		v := atomic.And64(&x, ^(1 << i))
		if r := uint64(0xffffffffffffffff) << (i + 1); x != r || v != old {
			t.Fatalf("clearing bit %#x: want %#x, got new %#x and old %#v", uint64(1<<i), r, x, v)
		}
	}

	// Set every bit in array to 1.
	a := make([]uint64, 1<<12)
	for i := range a {
		a[i] = 0xffffffffffffffff
	}

	// Clear array bit-by-bit in different goroutines.
	done := make(chan bool)
	for i := 0; i < 64; i++ {
		m := ^uint64(1 << i)
		go func() {
			for i := range a {
				atomic.And64(&a[i], m)
			}
			done <- true
		}()
	}
	for i := 0; i < 64; i++ {
		<-done
	}

	// Check that the array has been totally cleared.
	for i, v := range a {
		if v != 0 {
			t.Fatalf("a[%v] not cleared: want %#x, got %#x", i, uint64(0), v)
		}
	}
}

func TestOr32(t *testing.T) {
	// Basic sanity check.
	x := uint32(0)
	for i := uint32(0); i < 32; i++ {
		old := x
		v := atomic.Or32(&x, 1<<i)
		if r := (uint32(1) << (i + 1)) - 1; x != r || v != old {
			t.Fatalf("setting bit %#x: want %#x, got new %#x and old %#v", uint32(1<<i), r, x, v)
		}
	}

	// Start with every bit in array set to 0.
	a := make([]uint32, 1<<12)

	// Set every bit in array bit-by-bit in different goroutines.
	done := make(chan bool)
	for i := 0; i < 32; i++ {
		m := uint32(1 << i)
		go func() {
			for i := range a {
				atomic.Or32(&a[i], m)
			}
			done <- true
		}()
	}
	for i := 0; i < 32; i++ {
		<-done
	}

	// Check that the array has been totally set.
	for i, v := range a {
		if v != 0xffffffff {
			t.Fatalf("a[%v] not fully set: want %#x, got %#x", i, uint32(0xffffffff), v)
		}
	}
}

func TestOr64(t *testing.T) {
	// Basic sanity check.
	x := uint64(0)
	sink = &x
	for i := uint64(0); i < 64; i++ {
		old := x
		v := atomic.Or64(&x, 1<<i)
		if r := (uint64(1) << (i + 1)) - 1; x != r || v != old {
			t.Fatalf("setting bit %#x: want %#x, got new %#x and old %#v", uint64(1<<i), r, x, v)
		}
	}

	// Start with every bit in array set to 0.
	a := make([]uint64, 1<<12)

	// Set every bit in array bit-by-bit in different goroutines.
	done := make(chan bool)
	for i := 0; i < 64; i++ {
		m := uint64(1 << i)
		go func() {
			for i := range a {
				atomic.Or64(&a[i], m)
			}
			done <- true
		}()
	}
	for i := 0; i < 64; i++ {
		<-done
	}

	// Check that the array has been totally set.
	for i, v := range a {
		if v != 0xffffffffffffffff {
			t.Fatalf("a[%v] not fully set: want %#x, got %#x", i, uint64(0xffffffffffffffff), v)
		}
	}
}

func BenchmarkAnd32(b *testing.B) {
	var x [128]uint32 // give x its own cache line
	sink = &x
	for i := 0; i < b.N; i++ {
		atomic.And32(&x[63], uint32(i))
	}
}

func BenchmarkAnd32Parallel(b *testing.B) {
	var x [128]uint32 // give x its own cache line
	sink = &x
	b.RunParallel(func(pb *testing.PB) {
		i := uint32(0)
		for pb.Next() {
			atomic.And32(&x[63], i)
			i++
		}
	})
}

func BenchmarkAnd64(b *testing.B) {
	var x [128]uint64 // give x its own cache line
	sink = &x
	for i := 0; i < b.N; i++ {
		atomic.And64(&x[63], uint64(i))
	}
}

func BenchmarkAnd64Parallel(b *testing.B) {
	var x [128]uint64 // give x its own cache line
	sink = &x
	b.RunParallel(func(pb *testing.PB) {
		i := uint64(0)
		for pb.Next() {
			atomic.And64(&x[63], i)
			i++
		}
	})
}

func BenchmarkOr32(b *testing.B) {
	var x [128]uint32 // give x its own cache line
	sink = &x
	for i := 0; i < b.N; i++ {
		atomic.Or32(&x[63], uint32(i))
	}
}

func BenchmarkOr32Parallel(b *testing.B) {
	var x [128]uint32 // give x its own cache line
	sink = &x
	b.RunParallel(func(pb *testing.PB) {
		i := uint32(0)
		for pb.Next() {
			atomic.Or32(&x[63], i)
			i++
		}
	})
}

func BenchmarkOr64(b *testing.B) {
	var x [128]uint64 // give x its own cache line
	sink = &x
	for i := 0; i < b.N; i++ {
		atomic.Or64(&x[63], uint64(i))
	}
}

func BenchmarkOr64Parallel(b *testing.B) {
	var x [128]uint64 // give x its own cache line
	sink = &x
	b.RunParallel(func(pb *testing.PB) {
		i := uint64(0)
		for pb.Next() {
			atomic.Or64(&x[63], i)
			i++
		}
	})
}
