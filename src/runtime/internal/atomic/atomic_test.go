// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	"runtime"
	"runtime/internal/atomic"
	"testing"
	"unsafe"
)

func runParallel(N, iter int, f func()) {
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(int(N)))
	done := make(chan bool)
	for i := 0; i < N; i++ {
		go func() {
			for j := 0; j < iter; j++ {
				f()
			}
			done <- true
		}()
	}
	for i := 0; i < N; i++ {
		<-done
	}
}

func TestXadduintptr(t *testing.T) {
	const N = 20
	const iter = 100000
	inc := uintptr(100)
	total := uintptr(0)
	runParallel(N, iter, func() {
		atomic.Xadduintptr(&total, inc)
	})
	if want := uintptr(N * iter * inc); want != total {
		t.Fatalf("xadduintpr error, want %d, got %d", want, total)
	}
	total = 0
	runParallel(N, iter, func() {
		atomic.Xadduintptr(&total, inc)
		atomic.Xadduintptr(&total, uintptr(-int64(inc)))
	})
	if total != 0 {
		t.Fatalf("xadduintpr total error, want %d, got %d", 0, total)
	}
}

// Tests that xadduintptr correctly updates 64-bit values. The place where
// we actually do so is mstats.go, functions mSysStat{Inc,Dec}.
func TestXadduintptrOnUint64(t *testing.T) {
	/*	if runtime.BigEndian != 0 {
		// On big endian architectures, we never use xadduintptr to update
		// 64-bit values and hence we skip the test.  (Note that functions
		// mSysStat{Inc,Dec} in mstats.go have explicit checks for
		// big-endianness.)
		return
	}*/
	const inc = 100
	val := uint64(0)
	atomic.Xadduintptr((*uintptr)(unsafe.Pointer(&val)), inc)
	if inc != val {
		t.Fatalf("xadduintptr should increase lower-order bits, want %d, got %d", inc, val)
	}
}
