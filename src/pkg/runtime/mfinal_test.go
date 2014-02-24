// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
	"time"
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

	for _, tt := range finalizerTests {
		done := make(chan bool, 1)
		go func() {
			v := new(int)
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
			t.Errorf("finalizer for type %T didn't run", tt.finalizer)
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
