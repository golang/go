// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The race detector does not understand ParFor synchronization.
// +build !race

package runtime_test

import (
	. "runtime"
	"testing"
	"unsafe"
)

// Simple serial sanity test for parallelfor.
func TestParFor(t *testing.T) {
	const P = 1
	const N = 20
	data := make([]uint64, N)
	for i := uint64(0); i < N; i++ {
		data[i] = i
	}
	desc := NewParFor(P)
	ParForSetup(desc, P, N, nil, true, func(desc *ParFor, i uint32) {
		data[i] = data[i]*data[i] + 1
	})
	ParForDo(desc)
	for i := uint64(0); i < N; i++ {
		if data[i] != i*i+1 {
			t.Fatalf("Wrong element %d: %d", i, data[i])
		}
	}
}

// Test that nonblocking parallelfor does not block.
func TestParFor2(t *testing.T) {
	const P = 7
	const N = 1003
	data := make([]uint64, N)
	for i := uint64(0); i < N; i++ {
		data[i] = i
	}
	desc := NewParFor(P)
	ParForSetup(desc, P, N, (*byte)(unsafe.Pointer(&data)), false, func(desc *ParFor, i uint32) {
		d := *(*[]uint64)(unsafe.Pointer(desc.Ctx))
		d[i] = d[i]*d[i] + 1
	})
	for p := 0; p < P; p++ {
		ParForDo(desc)
	}
	for i := uint64(0); i < N; i++ {
		if data[i] != i*i+1 {
			t.Fatalf("Wrong element %d: %d", i, data[i])
		}
	}
}

// Test that iterations are properly distributed.
func TestParForSetup(t *testing.T) {
	const P = 11
	const N = 101
	desc := NewParFor(P)
	for n := uint32(0); n < N; n++ {
		for p := uint32(1); p <= P; p++ {
			ParForSetup(desc, p, n, nil, true, func(desc *ParFor, i uint32) {})
			sum := uint32(0)
			size0 := uint32(0)
			end0 := uint32(0)
			for i := uint32(0); i < p; i++ {
				begin, end := ParForIters(desc, i)
				size := end - begin
				sum += size
				if i == 0 {
					size0 = size
					if begin != 0 {
						t.Fatalf("incorrect begin: %d (n=%d, p=%d)", begin, n, p)
					}
				} else {
					if size != size0 && size != size0+1 {
						t.Fatalf("incorrect size: %d/%d (n=%d, p=%d)", size, size0, n, p)
					}
					if begin != end0 {
						t.Fatalf("incorrect begin/end: %d/%d (n=%d, p=%d)", begin, end0, n, p)
					}
				}
				end0 = end
			}
			if sum != n {
				t.Fatalf("incorrect sum: %d/%d (p=%d)", sum, n, p)
			}
		}
	}
}

// Test parallel parallelfor.
func TestParForParallel(t *testing.T) {
	if GOARCH != "amd64" {
		t.Log("temporarily disabled, see http://golang.org/issue/4155")
		return
	}

	N := uint64(1e7)
	if testing.Short() {
		N /= 10
	}
	data := make([]uint64, N)
	for i := uint64(0); i < N; i++ {
		data[i] = i
	}
	P := GOMAXPROCS(-1)
	c := make(chan bool, P)
	desc := NewParFor(uint32(P))
	ParForSetup(desc, uint32(P), uint32(N), nil, false, func(desc *ParFor, i uint32) {
		data[i] = data[i]*data[i] + 1
	})
	for p := 1; p < P; p++ {
		go func() {
			ParForDo(desc)
			c <- true
		}()
	}
	ParForDo(desc)
	for p := 1; p < P; p++ {
		<-c
	}
	for i := uint64(0); i < N; i++ {
		if data[i] != i*i+1 {
			t.Fatalf("Wrong element %d: %d", i, data[i])
		}
	}

	data, desc = nil, nil
	GC()
}
