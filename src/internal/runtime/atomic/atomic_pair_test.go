// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	"internal/runtime/atomic"
	"testing"
	"unsafe"
)

func TestCas128(t *testing.T) {
	var p atomic.Uint64Pair

	// Successful CAS from (0, 0) to (1, 2).
	if !p.CompareAndSwap(0, 0, 1, 2) {
		t.Fatal("Uint64Pair.CompareAndSwap: should have succeeded from zero")
	}
	if pair := p.Addr(); pair[0] != 1 || pair[1] != 2 {
		t.Fatalf("Uint64Pair.CompareAndSwap corrupt write: got (%d, %d), want (1, 2)", pair[0], pair[1])
	}

	// Mismatch on low half: should fail without writing.
	if p.CompareAndSwap(0, 2, 9, 9) {
		t.Fatal("Uint64Pair.CompareAndSwap: should have failed on low-half mismatch")
	}
	// Mismatch on high half: should fail without writing.
	if p.CompareAndSwap(1, 0, 9, 9) {
		t.Fatal("Uint64Pair.CompareAndSwap: should have failed on high-half mismatch")
	}
	if pair := p.Addr(); pair[0] != 1 || pair[1] != 2 {
		t.Fatalf("Uint64Pair.CompareAndSwap wrote on failed CAS: got (%d, %d), want (1, 2)", pair[0], pair[1])
	}

	// Concurrent test: 32 goroutines each bump (lo, hi) -> (lo+1, hi-1)
	// 1000 times. The invariant lo + hi == initialHi holds iff every
	// successful CAS updated both halves together.
	const initialHi = uint64(0xdeadbeefcafebabe)
	pair := p.Addr()
	pair[0], pair[1] = 0, initialHi

	const G, N = 32, 1000
	done := make(chan struct{})
	for g := 0; g < G; g++ {
		go func() {
			for i := 0; i < N; i++ {
				for {
					lo := atomic.Load64(&p.Addr()[0])
					hi := atomic.Load64(&p.Addr()[1])
					if p.CompareAndSwap(lo, hi, lo+1, hi-1) {
						break
					}
				}
			}
			done <- struct{}{}
		}()
	}
	for g := 0; g < G; g++ {
		<-done
	}
	if got, want := atomic.Load64(&p.Addr()[0]), uint64(G*N); got != want {
		t.Errorf("low half: got %d, want %d", got, want)
	}
	if got, want := atomic.Load64(&p.Addr()[1]), initialHi-uint64(G*N); got != want {
		t.Errorf("high half: got %#x, want %#x", got, want)
	}
}

func TestCas128Unaligned(t *testing.T) {
	// Bypass the struct's alignment guarantee to exercise the assembly
	// safety check. Use a buffer of 3 uint64s and pick a slot that is
	// 8-byte aligned but not 16-byte aligned.
	var buf [3]uint64
	var ptr uintptr
	if uintptr(unsafe.Pointer(&buf[0]))&15 == 0 {
		ptr = uintptr(unsafe.Pointer(&buf[1]))
	} else {
		ptr = uintptr(unsafe.Pointer(&buf[0]))
	}
	p := (*atomic.Uint64Pair)(unsafe.Pointer(ptr))

	defer func() {
		err := recover()
		const want = "unaligned 128-bit atomic operation"
		if err == nil {
			t.Fatal("Uint64Pair.CompareAndSwap on misaligned address did not panic")
		}
		if s, _ := err.(string); s != want {
			t.Fatalf("Uint64Pair.CompareAndSwap: got panic %q, want %q", err, want)
		}
	}()
	p.CompareAndSwap(0, 0, 0, 0)
}

func BenchmarkCas128(b *testing.B) {
	var p atomic.Uint64Pair
	for i := 0; i < b.N; i++ {
		p.CompareAndSwap(0, 0, 0, 0)
	}
}

func BenchmarkCas128Parallel(b *testing.B) {
	var p atomic.Uint64Pair
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			p.CompareAndSwap(0, 0, 0, 0)
		}
	})
}
