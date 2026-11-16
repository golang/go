// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime_test

import (
	"runtime"
	"testing"
	"unsafe"
)

func checkLogosScanWork(t *testing.T, wantRegion uintptr) {
	t.Helper()
	work, stackSample := runtime.LogosArenaScanWork()
	const wantStack = 8 * unsafe.Sizeof(uintptr(0))
	if stackSample != uint64(wantStack) {
		t.Errorf("physical stack sample = %d, want %d", stackSample, wantStack)
	}
	if want := int64(wantStack + wantRegion); work != want {
		t.Errorf("scan work = %d, want %d (%d stack + %d region)", work, want, wantStack, wantRegion)
	}
}

func TestLogosArenaScanWork(t *testing.T) {
	const word = unsafe.Sizeof(uintptr(0))
	type tail struct {
		p *byte
		x [7]uintptr
	}
	type sparse struct {
		p *byte
		x [65]uintptr
		q *byte
		y [3]uintptr
	}
	for _, tc := range []struct {
		name string
		elem any
		size uintptr
		want uintptr
	}{
		{"empty", (*byte)(nil), 0, 0},
		{"nil_pointer", (*byte)(nil), word, word},
		{"two_nil_pointers", (*byte)(nil), 2 * word, 2 * word},
		{"scalar_tail", tail{}, unsafe.Sizeof(tail{}), word},
		{"repeated_scalar_tail", tail{}, 3 * unsafe.Sizeof(tail{}), 17 * word},
		{"multiple_mask_words", sparse{}, unsafe.Sizeof(sparse{}), 67 * word},
		{"repeated_mask_words", sparse{}, 2 * unsafe.Sizeof(sparse{}), 137 * word},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inFreshGoroutine(t, func() {
				mark := runtime.LogosArenaSave()
				defer runtime.LogosArenaRestore(mark)
				runtime.LogosArenaAllocScan(tc.elem, tc.size)
				checkLogosScanWork(t, tc.want)
				runtime.LogosArenaRestore(mark)
				checkLogosScanWork(t, 0)
			})
		})
	}
}

func TestLogosArenaScanWorkChunks(t *testing.T) {
	inFreshGoroutine(t, func() {
		mark := runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(mark)
		runtime.LogosArenaAlloc(byte(0), 128)
		checkLogosScanWork(t, 0)
		const payload = runtime.LogosArenaChunkBytes / 2
		for range 3 {
			runtime.LogosArenaAllocScan((*byte)(nil), payload)
		}
		if chunks := runtime.LogosArenaStat().Chunks; chunks < 2 {
			t.Fatalf("fixture uses %d chunks, want at least 2", chunks)
		}
		checkLogosScanWork(t, 3*payload)
	})
}

func TestLogosRegionScanWork(t *testing.T) {
	inFreshGoroutine(t, func() {
		const word = unsafe.Sizeof(uintptr(0))
		mark := runtime.LogosArenaSave()
		defer runtime.LogosArenaRestore(mark)
		runtime.LogosArenaAllocScan((*byte)(nil), word)

		outer, outerGen := runtime.LogosRegionOpen()
		defer runtime.LogosRegionClose(outer, outerGen)
		runtime.LogosRegionAllocNoscan(outer, outerGen, 128, word)
		checkLogosScanWork(t, word)
		runtime.LogosRegionAllocScan(outer, outerGen, (*byte)(nil), 2*word)
		checkLogosScanWork(t, 3*word)

		inner, innerGen := runtime.LogosRegionOpen()
		runtime.LogosRegionAllocScan(inner, innerGen, (*byte)(nil), 3*word)
		checkLogosScanWork(t, 6*word)
		runtime.LogosRegionClose(inner, innerGen)
		checkLogosScanWork(t, 3*word)
	})
}
