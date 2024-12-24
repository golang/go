// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestCleanup(t *testing.T) {
	ch := make(chan bool, 1)
	done := make(chan bool, 1)
	want := 97531
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
		cleanup := func(x int) {
			if x != want {
				t.Errorf("cleanup %d, want %d", x, want)
			}
			ch <- true
		}
		runtime.AddCleanup(v, cleanup, 97531)
		v = nil
		done <- true
	}()
	<-done
	runtime.GC()
	<-ch
}

func TestCleanupMultiple(t *testing.T) {
	ch := make(chan bool, 3)
	done := make(chan bool, 1)
	want := 97531
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
		cleanup := func(x int) {
			if x != want {
				t.Errorf("cleanup %d, want %d", x, want)
			}
			ch <- true
		}
		runtime.AddCleanup(v, cleanup, 97531)
		runtime.AddCleanup(v, cleanup, 97531)
		runtime.AddCleanup(v, cleanup, 97531)
		v = nil
		done <- true
	}()
	<-done
	runtime.GC()
	<-ch
	<-ch
	<-ch
}

func TestCleanupZeroSizedStruct(t *testing.T) {
	type Z struct{}
	z := new(Z)
	runtime.AddCleanup(z, func(s string) {}, "foo")
}

func TestCleanupAfterFinalizer(t *testing.T) {
	ch := make(chan int, 2)
	done := make(chan bool, 1)
	want := 97531
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
		finalizer := func(x *int) {
			ch <- 1
		}
		cleanup := func(x int) {
			if x != want {
				t.Errorf("cleanup %d, want %d", x, want)
			}
			ch <- 2
		}
		runtime.AddCleanup(v, cleanup, 97531)
		runtime.SetFinalizer(v, finalizer)
		v = nil
		done <- true
	}()
	<-done
	runtime.GC()
	var result int
	result = <-ch
	if result != 1 {
		t.Errorf("result %d, want 1", result)
	}
	runtime.GC()
	result = <-ch
	if result != 2 {
		t.Errorf("result %d, want 2", result)
	}
}

func TestCleanupInteriorPointer(t *testing.T) {
	ch := make(chan bool, 3)
	done := make(chan bool, 1)
	want := 97531
	go func() {
		// Allocate struct with pointer to avoid hitting tinyalloc.
		// Otherwise we can't be sure when the allocation will
		// be freed.
		type T struct {
			p unsafe.Pointer
			i int
			a int
			b int
			c int
		}
		ts := new(T)
		ts.a = 97531
		ts.b = 97531
		ts.c = 97531
		cleanup := func(x int) {
			if x != want {
				t.Errorf("cleanup %d, want %d", x, want)
			}
			ch <- true
		}
		runtime.AddCleanup(&ts.a, cleanup, 97531)
		runtime.AddCleanup(&ts.b, cleanup, 97531)
		runtime.AddCleanup(&ts.c, cleanup, 97531)
		ts = nil
		done <- true
	}()
	<-done
	runtime.GC()
	<-ch
	<-ch
	<-ch
}

func TestCleanupStop(t *testing.T) {
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
		cleanup := func(x int) {
			t.Error("cleanup called, want no cleanup called")
		}
		c := runtime.AddCleanup(v, cleanup, 97531)
		c.Stop()
		v = nil
		done <- true
	}()
	<-done
	runtime.GC()
}

func TestCleanupStopMultiple(t *testing.T) {
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
		cleanup := func(x int) {
			t.Error("cleanup called, want no cleanup called")
		}
		c := runtime.AddCleanup(v, cleanup, 97531)
		c.Stop()
		c.Stop()
		c.Stop()
		v = nil
		done <- true
	}()
	<-done
	runtime.GC()
}

func TestCleanupStopinterleavedMultiple(t *testing.T) {
	ch := make(chan bool, 3)
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
		cleanup := func(x int) {
			if x != 1 {
				t.Error("cleanup called, want no cleanup called")
			}
			ch <- true
		}
		runtime.AddCleanup(v, cleanup, 1)
		runtime.AddCleanup(v, cleanup, 2).Stop()
		runtime.AddCleanup(v, cleanup, 1)
		runtime.AddCleanup(v, cleanup, 2).Stop()
		runtime.AddCleanup(v, cleanup, 1)
		v = nil
		done <- true
	}()
	<-done
	runtime.GC()
	<-ch
	<-ch
	<-ch
}

func TestCleanupStopAfterCleanupRuns(t *testing.T) {
	ch := make(chan bool, 1)
	done := make(chan bool, 1)
	var stop func()
	go func() {
		// Allocate struct with pointer to avoid hitting tinyalloc.
		// Otherwise we can't be sure when the allocation will
		// be freed.
		type T struct {
			v int
			p unsafe.Pointer
		}
		v := &new(T).v
		*v = 97531
		cleanup := func(x int) {
			ch <- true
		}
		cl := runtime.AddCleanup(v, cleanup, 97531)
		v = nil
		stop = cl.Stop
		done <- true
	}()
	<-done
	runtime.GC()
	<-ch
	stop()
}
