// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
	"unsafe"
)

func TestMemStats(t *testing.T) {
	// Test that MemStats has sane values.
	st := new(MemStats)
	ReadMemStats(st)
	if st.HeapSys == 0 || st.StackSys == 0 || st.MSpanSys == 0 || st.MCacheSys == 0 ||
		st.BuckHashSys == 0 || st.GCSys == 0 || st.OtherSys == 0 {
		t.Fatalf("Zero sys value: %+v", *st)
	}
	if st.Sys != st.HeapSys+st.StackSys+st.MSpanSys+st.MCacheSys+
		st.BuckHashSys+st.GCSys+st.OtherSys {
		t.Fatalf("Bad sys value: %+v", *st)
	}
}

var mallocSink uintptr

func BenchmarkMalloc8(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := new(int64)
		x ^= uintptr(unsafe.Pointer(p))
	}
	mallocSink = x
}

func BenchmarkMalloc16(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := new([2]int64)
		x ^= uintptr(unsafe.Pointer(p))
	}
	mallocSink = x
}

func BenchmarkMallocTypeInfo8(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := new(struct {
			p [8 / unsafe.Sizeof(uintptr(0))]*int
		})
		x ^= uintptr(unsafe.Pointer(p))
	}
	mallocSink = x
}

func BenchmarkMallocTypeInfo16(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := new(struct {
			p [16 / unsafe.Sizeof(uintptr(0))]*int
		})
		x ^= uintptr(unsafe.Pointer(p))
	}
	mallocSink = x
}
