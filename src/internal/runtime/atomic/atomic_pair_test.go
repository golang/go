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
	if lo, hi := p.Load(); lo != 1 || hi != 2 {
		t.Fatalf("Uint64Pair.CompareAndSwap corrupt write: got (%d, %d), want (1, 2)", lo, hi)
	}

	// Mismatch on low half: should fail without writing.
	if p.CompareAndSwap(0, 2, 9, 9) {
		t.Fatal("Uint64Pair.CompareAndSwap: should have failed on low-half mismatch")
	}
	// Mismatch on high half: should fail without writing.
	if p.CompareAndSwap(1, 0, 9, 9) {
		t.Fatal("Uint64Pair.CompareAndSwap: should have failed on high-half mismatch")
	}
	if lo, hi := p.Load(); lo != 1 || hi != 2 {
		t.Fatalf("Uint64Pair.CompareAndSwap wrote on failed CAS: got (%d, %d), want (1, 2)", lo, hi)
	}

	// Concurrent test: 32 goroutines each bump (lo, hi) -> (lo+1, hi-1)
	// 1000 times. The invariant lo + hi == initialHi holds iff every
	// successful CAS updated both halves together.
	const initialHi = uint64(0xdeadbeefcafebabe)
	p.Store(0, initialHi)

	const G, N = 32, 1000
	done := make(chan struct{})
	for g := 0; g < G; g++ {
		go func() {
			for i := 0; i < N; i++ {
				for {
					lo, hi := p.Load()
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
	if lo, hi := p.Load(); lo != uint64(G*N) {
		t.Errorf("low half: got %d, want %d", lo, uint64(G*N))
	} else if hi != initialHi-uint64(G*N) {
		t.Errorf("high half: got %#x, want %#x", hi, initialHi-uint64(G*N))
	}
}

func TestCas128Unaligned(t *testing.T) {
	// Bypass the struct's alignment guarantee to exercise the assembly
	// safety check. Use a buffer of 3 uint64s and pick a slot that is
	// 8-byte aligned but not 16-byte aligned.
	var buf [3]uint64
	var ptr unsafe.Pointer
	if uintptr(unsafe.Pointer(&buf[0]))&15 == 0 {
		ptr = unsafe.Pointer(&buf[1])
	} else {
		ptr = unsafe.Pointer(&buf[0])
	}
	p := (*atomic.Uint64Pair)(ptr)

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
