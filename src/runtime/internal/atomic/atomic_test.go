// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package atomic_test

import (
	"runtime"
	"runtime/internal/atomic"
	"runtime/internal/sys"
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
	if sys.BigEndian {
		// On big endian architectures, we never use xadduintptr to update
		// 64-bit values and hence we skip the test.  (Note that functions
		// mSysStat{Inc,Dec} in mstats.go have explicit checks for
		// big-endianness.)
		t.Skip("skip xadduintptr on big endian architecture")
	}
	const inc = 100
	val := uint64(0)
	atomic.Xadduintptr((*uintptr)(unsafe.Pointer(&val)), inc)
	if inc != val {
		t.Fatalf("xadduintptr should increase lower-order bits, want %d, got %d", inc, val)
	}
}

func shouldPanic(t *testing.T, name string, f func()) {
	defer func() {
		if recover() == nil {
			t.Errorf("%s did not panic", name)
		}
	}()
	f()
}

// Variant of sync/atomic's TestUnaligned64:
func TestUnaligned64(t *testing.T) {
	// Unaligned 64-bit atomics on 32-bit systems are
	// a continual source of pain. Test that on 32-bit systems they crash
	// instead of failing silently.

	switch runtime.GOARCH {
	default:
		if unsafe.Sizeof(int(0)) != 4 {
			t.Skip("test only runs on 32-bit systems")
		}
	case "amd64p32":
		// amd64p32 can handle unaligned atomics.
		t.Skipf("test not needed on %v", runtime.GOARCH)
	}

	x := make([]uint32, 4)
	u := unsafe.Pointer(uintptr(unsafe.Pointer(&x[0])) | 4) // force alignment to 4

	up64 := (*uint64)(u) // misaligned
	p64 := (*int64)(u)   // misaligned

	shouldPanic(t, "Load64", func() { atomic.Load64(up64) })
	shouldPanic(t, "Loadint64", func() { atomic.Loadint64(p64) })
	shouldPanic(t, "Store64", func() { atomic.Store64(up64, 0) })
	shouldPanic(t, "Xadd64", func() { atomic.Xadd64(up64, 1) })
	shouldPanic(t, "Xchg64", func() { atomic.Xchg64(up64, 1) })
	shouldPanic(t, "Cas64", func() { atomic.Cas64(up64, 1, 2) })
}
