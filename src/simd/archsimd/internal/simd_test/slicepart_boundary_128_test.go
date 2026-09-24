// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd && (amd64 || arm64) && linux

package simd_test

import (
	"fmt"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"simd/archsimd"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

// The tests in this file check that the Load*Part functions access only the
// elements of their slice argument. They place the slice at the start and at
// the end of a page that lies between two inaccessible pages, so that any
// access outside the slice faults. See also bytes/boundary_test.go.

func TestLoadPartPageBoundary128(t *testing.T) {
	testLoadPartPageBoundary(t, archsimd.LoadInt8x16Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt16x8Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt32x4Part)
	testLoadPartPageBoundary(t, archsimd.LoadInt64x2Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint8x16Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint16x8Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint32x4Part)
	testLoadPartPageBoundary(t, archsimd.LoadUint64x2Part)
	testLoadPartPageBoundary(t, archsimd.LoadFloat32x4Part)
	testLoadPartPageBoundary(t, archsimd.LoadFloat64x2Part)
}

// HasLenAndStore is implemented by the vector types returned by the Load*Part
// functions.
type HasLenAndStore[T number] interface {
	Len() int
	Store(s []T)
}

// testLoadPartPageBoundary runs a subtest that checks that load, a Load*Part
// function, reads only the elements of its argument and loads them correctly.
// It tries every slice length from 1 to the vector length, with the slice at
// each end of a page whose neighbors are inaccessible.
func testLoadPartPageBoundary[T number, V HasLenAndStore[T]](t *testing.T, load func(s []T) (V, int)) {
	name := runtime.FuncForPC(reflect.ValueOf(load).Pointer()).Name()
	name = name[strings.LastIndexByte(name, '.')+1:]
	t.Run(name, func(t *testing.T) {
		var zero V
		n := zero.Len()
		size := int(unsafe.Sizeof(T(0)))
		// TODO: Remove this once the amd64 Part loads that are generated from
		// the masked-load templates (those for 32- and 64-bit elements, and
		// all 512-bit ones) stop loading the whole vector.
		if runtime.GOARCH == "amd64" && (size >= 4 || n*size == 64) {
			t.Skipf("skipping: %s is known to read past the end of its slice", name)
		}
		page := guardedPage(t)
		for l := 1; l <= n; l++ {
			for _, off := range []int{0, len(page) - l*size} {
				s := unsafe.Slice((*T)(unsafe.Pointer(&page[off])), l)
				for i := range s {
					s[i] = T(i + 1)
				}
				v, got, fault := loadCatchingFault(load, s)
				if fault != nil {
					start := uintptr(unsafe.Pointer(&s[0]))
					where := "past the end of"
					if fault.addr < start {
						where = "before the start of"
					}
					t.Fatalf("%s read %s a %d-element slice at [%#x, %#x): fault at %#x in %s",
						name, where, l, start, start+uintptr(l*size), fault.addr, fault.loc)
				}
				if got != l {
					t.Errorf("%s(s) with len(s) = %d returned %d, want %d", name, l, got, l)
				}
				gotElems := make([]T, n)
				v.Store(gotElems)
				wantElems := make([]T, n)
				copy(wantElems, s)
				checkSlicesLogInput(t, gotElems, wantElems, 0.0, func() { t.Helper(); t.Logf("len(s) = %d", l) })
			}
		}
	})
}

// guardedPage returns a page of memory that is immediately preceded and
// followed by inaccessible pages.
func guardedPage(t *testing.T) []byte {
	t.Helper()
	size := syscall.Getpagesize()
	mem, err := syscall.Mmap(-1, 0, 3*size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap failed: %v", err)
	}
	t.Cleanup(func() {
		if err := syscall.Munmap(mem); err != nil {
			t.Errorf("munmap failed: %v", err)
		}
	})
	if err := syscall.Mprotect(mem[:size], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect of low page failed: %v", err)
	}
	if err := syscall.Mprotect(mem[2*size:], syscall.PROT_NONE); err != nil {
		t.Fatalf("mprotect of high page failed: %v", err)
	}
	return mem[size : 2*size : 2*size]
}

// A faultInfo describes a memory fault that the runtime turned into a panic.
type faultInfo struct {
	addr uintptr // the address whose access faulted
	loc  string  // the function and source line that made the access
}

// loadCatchingFault returns load(s). If load faults, it returns a description
// of the fault instead of crashing.
func loadCatchingFault[T number, V any](load func(s []T) (V, int), s []T) (v V, n int, fault *faultInfo) {
	old := debug.SetPanicOnFault(true)
	defer debug.SetPanicOnFault(old)
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		err, ok := r.(interface{ Addr() uintptr })
		if !ok {
			panic(r)
		}
		fault = &faultInfo{addr: err.Addr(), loc: faultLocation()}
	}()
	v, n = load(s)
	return v, n, nil
}

// faultLocation describes the memory access that caused the panic in
// progress: the function and source line that made it, followed by the
// callers inside package archsimd, which end with the function that the test
// called. A deferred function must call faultLocation while that panic is
// being handled, when the faulting frame is still on the stack.
func faultLocation() string {
	pcs := make([]uintptr, 64)
	frames := runtime.CallersFrames(pcs[:runtime.Callers(1, pcs)])
	// Skip to runtime.sigpanic. The frame below it made the access.
	for {
		frame, more := frames.Next()
		if !more {
			return "unknown location"
		}
		if frame.Function == "runtime.sigpanic" {
			break
		}
	}
	var locs []string
	for {
		frame, more := frames.Next()
		if len(locs) > 0 && !strings.HasPrefix(frame.Function, "simd/archsimd.") {
			break
		}
		locs = append(locs, fmt.Sprintf("%s (%s:%d)", frame.Function, filepath.Base(frame.File), frame.Line))
		if !more {
			break
		}
	}
	return strings.Join(locs, ", called from ")
}
