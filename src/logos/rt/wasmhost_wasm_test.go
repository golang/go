// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"errors"
	"runtime"
	"testing"
	"unsafe"
)

func bytesAt(p, n int32) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(uintptr(p))), n)
}

func TestScratchBuffersStayAliveUntilReset(t *testing.T) {
	a, b := scratchAlloc(32), scratchAlloc(32)
	if a == b {
		t.Fatalf("two live buffers share address %#x", a)
	}
	copy(bytesAt(a, 32), "alpha")
	copy(bytesAt(b, 32), "bravo")
	runtime.GC()
	runtime.GC()
	if got := string(bytesAt(a, 5)); got != "alpha" {
		t.Fatalf("buffer a holds %q after GC, want alpha", got)
	}
	if got := string(bytesAt(b, 5)); got != "bravo" {
		t.Fatalf("buffer b holds %q after GC, want bravo", got)
	}
	scratchReset(a)
	if _, live := scratch[a]; live {
		t.Fatal("scratchReset left the buffer in the live map")
	}
	if _, live := scratch[b]; !live {
		t.Fatal("scratchReset of a released the other buffer")
	}
	scratchReset(b)
	scratchReset(b)
	if len(scratch) != 0 {
		t.Fatalf("%d buffers live after resets", len(scratch))
	}
}

func TestScratchAllocIsZeroedAndSized(t *testing.T) {
	p := scratchAlloc(1 << 20)
	defer scratchReset(p)
	if len(scratch[p]) != 1<<20 {
		t.Fatalf("buffer length = %d, want %d", len(scratch[p]), 1<<20)
	}
	for i, v := range bytesAt(p, 1<<20) {
		if v != 0 {
			t.Fatalf("byte %d is %d, want 0", i, v)
		}
	}
}

func TestScratchAllocRejectsBadSize(t *testing.T) {
	before := len(scratch)
	if got := scratchAlloc(-1); got != scratchRefused {
		t.Fatalf("scratchAlloc(-1) = %#x, want the refusal address %d", got, scratchRefused)
	}
	if got := scratchAlloc(maxScratchBytes + 1); got != scratchRefused {
		t.Fatalf("scratchAlloc(limit+1) = %#x, want the refusal address %d", got, scratchRefused)
	}
	if got := scratchAlloc(1<<31 - 1); got != scratchRefused {
		t.Fatalf("scratchAlloc(maxint32) = %#x, want the refusal address %d", got, scratchRefused)
	}
	if len(scratch) != before {
		t.Fatal("rejected scratchAlloc changed the live map")
	}
	p := scratchAlloc(16)
	if p == scratchRefused {
		t.Fatal("scratchAlloc(16) was refused, want a live buffer")
	}
	defer scratchReset(p)
	if len(scratch[p]) != 16 {
		t.Fatalf("buffer length = %d, want 16", len(scratch[p]))
	}
}

func TestErrmsgExportsDescribeLastError(t *testing.T) {
	WasmError(errors.New("exact: text"))
	got := string(bytesAt(errmsgPtr(), errmsgLen()))
	if got != "exact: text" {
		t.Fatalf("memory at errmsg ptr/len = %q, want the exact error text", got)
	}
}
