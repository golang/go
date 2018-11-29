// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
	"time"
	"unsafe"
)

type Tintptr *int // assignable to *int
type Tint int     // *Tint implements Tinter, interface{}

func (t *Tint) m() {}

type Tinter interface {
	m()
}

func TestFinalizerType(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skipf("Skipping on non-amd64 machine")
	}

	ch := make(chan bool, 10)
	finalize := func(x *int) {
		if *x != 97531 {
			t.Errorf("finalizer %d, want %d", *x, 97531)
		}
		ch <- true
	}

	var finalizerTests = []struct {
		convert   func(*int) interface{}
		finalizer interface{}
	}{
		{func(x *int) interface{} { return x }, func(v *int) { finalize(v) }},
		{func(x *int) interface{} { return Tintptr(x) }, func(v Tintptr) { finalize(v) }},
		{func(x *int) interface{} { return Tintptr(x) }, func(v *int) { finalize(v) }},
		{func(x *int) interface{} { return (*Tint)(x) }, func(v *Tint) { finalize((*int)(v)) }},
		{func(x *int) interface{} { return (*Tint)(x) }, func(v Tinter) { finalize((*int)(v.(*Tint))) }},
	}

	for i, tt := range finalizerTests {
		done := make(chan bool, 1)
		go func() {
			// allocate struct with pointer to avoid hitting tinyalloc.
			// Otherwise we can't be sure when the allocation will
			// be freed.
			type T struct {
				v int
				p unsafe.Pointer
			}
			v := &new(T).v
			*v = 97531
			runtime.SetFinalizer(tt.convert(v), tt.finalizer)
			v = nil
			done <- true
		}()
		<-done
		runtime.GC()
		select {
		case <-ch:
		case <-time.After(time.Second * 4):
			t.Errorf("#%d: finalizer for type %T didn't run", i, tt.finalizer)
		}
	}
}

type bigValue struct {
	fill uint64
	it   bool
	up   string
}

func TestFinalizerInterfaceBig(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skipf("Skipping on non-amd64 machine")
	}
	ch := make(chan bool)
	done := make(chan bool, 1)
	go func() {
		v := &bigValue{0xDEADBEEFDEADBEEF, true, "It matters not how strait the gate"}
		old := *v
		runtime.SetFinalizer(v, func(v interface{}) {
			i, ok := v.(*bigValue)
			if !ok {
				t.Errorf("finalizer called with type %T, want *bigValue", v)
			}
			if *i != old {
				t.Errorf("finalizer called with %+v, want %+v", *i, old)
			}
			close(ch)
		})
		v = nil
		done <- true
	}()
	<-done
	runtime.GC()
	select {
	case <-ch:
	case <-time.After(4 * time.Second):
		t.Errorf("finalizer for type *bigValue didn't run")
	}
}

func fin(v *int) {
}

// Verify we don't crash at least. golang.org/issue/6857
func TestFinalizerZeroSizedStruct(t *testing.T) {
	type Z struct{}
	z := new(Z)
	runtime.SetFinalizer(z, func(*Z) {})
}

func BenchmarkFinalizer(b *testing.B) {
	const Batch = 1000
	b.RunParallel(func(pb *testing.PB) {
		var data [Batch]*int
		for i := 0; i < Batch; i++ {
			data[i] = new(int)
		}
		for pb.Next() {
			for i := 0; i < Batch; i++ {
				runtime.SetFinalizer(data[i], fin)
			}
			for i := 0; i < Batch; i++ {
				runtime.SetFinalizer(data[i], nil)
			}
		}
	})
}

func BenchmarkFinalizerRun(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v := new(int)
			runtime.SetFinalizer(v, fin)
		}
	})
}

// One chunk must be exactly one sizeclass in size.
// It should be a sizeclass not used much by others, so we
// have a greater chance of finding adjacent ones.
// size class 19: 320 byte objects, 25 per page, 1 page alloc at a time
const objsize = 320

type objtype [objsize]byte

func adjChunks() (*objtype, *objtype) {
	var s []*objtype

	for {
		c := new(objtype)
		for _, d := range s {
			if uintptr(unsafe.Pointer(c))+unsafe.Sizeof(*c) == uintptr(unsafe.Pointer(d)) {
				return c, d
			}
			if uintptr(unsafe.Pointer(d))+unsafe.Sizeof(*c) == uintptr(unsafe.Pointer(c)) {
				return d, c
			}
		}
		s = append(s, c)
	}
}

// Make sure an empty slice on the stack doesn't pin the next object in memory.
func TestEmptySlice(t *testing.T) {
	x, y := adjChunks()

	// the pointer inside xs points to y.
	xs := x[objsize:] // change objsize to objsize-1 and the test passes

	fin := make(chan bool, 1)
	runtime.SetFinalizer(y, func(z *objtype) { fin <- true })
	runtime.GC()
	select {
	case <-fin:
	case <-time.After(4 * time.Second):
		t.Errorf("finalizer of next object in memory didn't run")
	}
	xsglobal = xs // keep empty slice alive until here
}

var xsglobal []byte

func adjStringChunk() (string, *objtype) {
	b := make([]byte, objsize)
	for {
		s := string(b)
		t := new(objtype)
		p := *(*uintptr)(unsafe.Pointer(&s))
		q := uintptr(unsafe.Pointer(t))
		if p+objsize == q {
			return s, t
		}
	}
}

// Make sure an empty string on the stack doesn't pin the next object in memory.
func TestEmptyString(t *testing.T) {
	x, y := adjStringChunk()

	ss := x[objsize:] // change objsize to objsize-1 and the test passes
	fin := make(chan bool, 1)
	// set finalizer on string contents of y
	runtime.SetFinalizer(y, func(z *objtype) { fin <- true })
	runtime.GC()
	select {
	case <-fin:
	case <-time.After(4 * time.Second):
		t.Errorf("finalizer of next string in memory didn't run")
	}
	ssglobal = ss // keep 0-length string live until here
}

var ssglobal string

// Test for issue 7656.
func TestFinalizerOnGlobal(t *testing.T) {
	runtime.SetFinalizer(Foo1, func(p *Object1) {})
	runtime.SetFinalizer(Foo2, func(p *Object2) {})
	runtime.SetFinalizer(Foo1, nil)
	runtime.SetFinalizer(Foo2, nil)
}

type Object1 struct {
	Something []byte
}

type Object2 struct {
	Something byte
}

var (
	Foo2 = &Object2{}
	Foo1 = &Object1{}
)

func TestDeferKeepAlive(t *testing.T) {
	if *flagQuick {
		t.Skip("-quick")
	}

	// See issue 21402.
	t.Parallel()
	type T *int // needs to be a pointer base type to avoid tinyalloc and its never-finalized behavior.
	x := new(T)
	finRun := false
	runtime.SetFinalizer(x, func(x *T) {
		finRun = true
	})
	defer runtime.KeepAlive(x)
	runtime.GC()
	time.Sleep(time.Second)
	if finRun {
		t.Errorf("finalizer ran prematurely")
	}
}
