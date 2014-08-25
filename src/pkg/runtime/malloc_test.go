// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"flag"
	. "runtime"
	"testing"
	"time"
	"unsafe"
)

func TestMemStats(t *testing.T) {
	// Test that MemStats has sane values.
	st := new(MemStats)
	ReadMemStats(st)

	// Everything except HeapReleased and HeapIdle, because they indeed can be 0.
	if st.Alloc == 0 || st.TotalAlloc == 0 || st.Sys == 0 || st.Lookups == 0 ||
		st.Mallocs == 0 || st.Frees == 0 || st.HeapAlloc == 0 || st.HeapSys == 0 ||
		st.HeapInuse == 0 || st.HeapObjects == 0 || st.StackInuse == 0 ||
		st.StackSys == 0 || st.MSpanInuse == 0 || st.MSpanSys == 0 || st.MCacheInuse == 0 ||
		st.MCacheSys == 0 || st.BuckHashSys == 0 || st.GCSys == 0 || st.OtherSys == 0 ||
		st.NextGC == 0 || st.NumGC == 0 {
		t.Fatalf("Zero value: %+v", *st)
	}

	if st.Alloc > 1e10 || st.TotalAlloc > 1e11 || st.Sys > 1e10 || st.Lookups > 1e10 ||
		st.Mallocs > 1e10 || st.Frees > 1e10 || st.HeapAlloc > 1e10 || st.HeapSys > 1e10 ||
		st.HeapIdle > 1e10 || st.HeapInuse > 1e10 || st.HeapObjects > 1e10 || st.StackInuse > 1e10 ||
		st.StackSys > 1e10 || st.MSpanInuse > 1e10 || st.MSpanSys > 1e10 || st.MCacheInuse > 1e10 ||
		st.MCacheSys > 1e10 || st.BuckHashSys > 1e10 || st.GCSys > 1e10 || st.OtherSys > 1e10 ||
		st.NextGC > 1e10 || st.NumGC > 1e9 {
		t.Fatalf("Insanely high value (overflow?): %+v", *st)
	}

	if st.Sys != st.HeapSys+st.StackSys+st.MSpanSys+st.MCacheSys+
		st.BuckHashSys+st.GCSys+st.OtherSys {
		t.Fatalf("Bad sys value: %+v", *st)
	}

	if st.HeapIdle+st.HeapInuse != st.HeapSys {
		t.Fatalf("HeapIdle(%d) + HeapInuse(%d) should be equal to HeapSys(%d), but isn't.", st.HeapIdle, st.HeapInuse, st.HeapSys)
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

type LargeStruct struct {
	x [16][]byte
}

func BenchmarkMallocLargeStruct(b *testing.B) {
	var x uintptr
	for i := 0; i < b.N; i++ {
		p := make([]LargeStruct, 2)
		x ^= uintptr(unsafe.Pointer(&p[0]))
	}
	mallocSink = x
}

var n = flag.Int("n", 1000, "number of goroutines")

func BenchmarkGoroutineSelect(b *testing.B) {
	quit := make(chan struct{})
	read := func(ch chan struct{}) {
		for {
			select {
			case _, ok := <-ch:
				if !ok {
					return
				}
			case <-quit:
				return
			}
		}
	}
	benchHelper(b, *n, read)
}

func BenchmarkGoroutineBlocking(b *testing.B) {
	read := func(ch chan struct{}) {
		for {
			if _, ok := <-ch; !ok {
				return
			}
		}
	}
	benchHelper(b, *n, read)
}

func BenchmarkGoroutineForRange(b *testing.B) {
	read := func(ch chan struct{}) {
		for range ch {
		}
	}
	benchHelper(b, *n, read)
}

func benchHelper(b *testing.B, n int, read func(chan struct{})) {
	m := make([]chan struct{}, n)
	for i := range m {
		m[i] = make(chan struct{}, 1)
		go read(m[i])
	}
	b.StopTimer()
	b.ResetTimer()
	GC()

	for i := 0; i < b.N; i++ {
		for _, ch := range m {
			if ch != nil {
				ch <- struct{}{}
			}
		}
		time.Sleep(10 * time.Millisecond)
		b.StartTimer()
		GC()
		b.StopTimer()
	}

	for _, ch := range m {
		close(ch)
	}
	time.Sleep(10 * time.Millisecond)
}

func BenchmarkGoroutineIdle(b *testing.B) {
	quit := make(chan struct{})
	fn := func() {
		<-quit
	}
	for i := 0; i < *n; i++ {
		go fn()
	}

	GC()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		GC()
	}

	b.StopTimer()
	close(quit)
	time.Sleep(10 * time.Millisecond)
}
