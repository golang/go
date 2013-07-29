// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestFinalizerTypeSucceed(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skipf("Skipping on non-amd64 machine")
	}
	ch := make(chan bool)
	func() {
		v := new(int)
		*v = 97531
		runtime.SetFinalizer(v, func(v *int) {
			if *v != 97531 {
				t.Errorf("*int in finalizer has the wrong value: %d\n", *v)
			}
			close(ch)
		})
		v = nil
	}()
	runtime.GC()
	select {
	case <-ch:
	case <-time.After(time.Second * 4):
		t.Errorf("Finalizer set by SetFinalizer(*int, func(*int)) didn't run")
	}
}

func TestFinalizerInterface(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skipf("Skipping on non-amd64 machine")
	}
	ch := make(chan bool)
	func() {
		v := new(int)
		*v = 97531
		runtime.SetFinalizer(v, func(v interface{}) {
			i, ok := v.(*int)
			if !ok {
				t.Errorf("Expected *int from interface{} in finalizer, got %v", *i)
			}
			if *i != 97531 {
				t.Errorf("*int from interface{} has the wrong value: %d\n", *i)
			}
			close(ch)
		})
		v = nil
	}()
	runtime.GC()
	select {
	case <-ch:
	case <-time.After(time.Second * 4):
		t.Errorf("Finalizer set by SetFinalizer(*int, func(interface{})) didn't run")
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
	func() {
		v := &bigValue{0xDEADBEEFDEADBEEF, true, "It matters not how strait the gate"}
		runtime.SetFinalizer(v, func(v interface{}) {
			i, ok := v.(*bigValue)
			if !ok {
				t.Errorf("Expected *bigValue from interface{} in finalizer, got %v", *i)
			}
			if i.fill != 0xDEADBEEFDEADBEEF && i.it != true && i.up != "It matters not how strait the gate" {
				t.Errorf("*bigValue from interface{} has the wrong value: %d\n", *i)
			}
			close(ch)
		})
		v = nil
	}()
	runtime.GC()
	select {
	case <-ch:
	case <-time.After(time.Second * 4):
		t.Errorf("Finalizer set by SetFinalizer(*bigValue, func(interface{})) didn't run")
	}
}

func fin(v *int) {
}

func BenchmarkFinalizer(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	var wg sync.WaitGroup
	wg.Add(procs)
	for p := 0; p < procs; p++ {
		go func() {
			var data [CallsPerSched]*int
			for i := 0; i < CallsPerSched; i++ {
				data[i] = new(int)
			}
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for i := 0; i < CallsPerSched; i++ {
					runtime.SetFinalizer(data[i], fin)
				}
				for i := 0; i < CallsPerSched; i++ {
					runtime.SetFinalizer(data[i], nil)
				}
			}
			wg.Done()
		}()
	}
	wg.Wait()
}

func BenchmarkFinalizerRun(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	var wg sync.WaitGroup
	wg.Add(procs)
	for p := 0; p < procs; p++ {
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for i := 0; i < CallsPerSched; i++ {
					v := new(int)
					runtime.SetFinalizer(v, fin)
				}
				runtime.GC()
			}
			wg.Done()
		}()
	}
	wg.Wait()
}
