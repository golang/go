// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

func TestNoRaceAtomicAddInt64(t *testing.T) {
	var x1, x2 int8
	var s int64
	ch := make(chan bool, 2)
	go func() {
		x1 = 1
		if atomic.AddInt64(&s, 1) == 2 {
			x2 = 1
		}
		ch <- true
	}()
	go func() {
		x2 = 1
		if atomic.AddInt64(&s, 1) == 2 {
			x1 = 1
		}
		ch <- true
	}()
	<-ch
	<-ch
}

func TestRaceAtomicAddInt64(t *testing.T) {
	var x1, x2 int8
	var s int64
	ch := make(chan bool, 2)
	go func() {
		x1 = 1
		if atomic.AddInt64(&s, 1) == 1 {
			x2 = 1
		}
		ch <- true
	}()
	go func() {
		x2 = 1
		if atomic.AddInt64(&s, 1) == 1 {
			x1 = 1
		}
		ch <- true
	}()
	<-ch
	<-ch
}

func TestNoRaceAtomicAddInt32(t *testing.T) {
	var x1, x2 int8
	var s int32
	ch := make(chan bool, 2)
	go func() {
		x1 = 1
		if atomic.AddInt32(&s, 1) == 2 {
			x2 = 1
		}
		ch <- true
	}()
	go func() {
		x2 = 1
		if atomic.AddInt32(&s, 1) == 2 {
			x1 = 1
		}
		ch <- true
	}()
	<-ch
	<-ch
}

func TestNoRaceAtomicLoadAddInt32(t *testing.T) {
	var x int64
	var s int32
	go func() {
		x = 2
		atomic.AddInt32(&s, 1)
	}()
	for atomic.LoadInt32(&s) != 1 {
		runtime.Gosched()
	}
	x = 1
}

func TestNoRaceAtomicLoadStoreInt32(t *testing.T) {
	var x int64
	var s int32
	go func() {
		x = 2
		atomic.StoreInt32(&s, 1)
	}()
	for atomic.LoadInt32(&s) != 1 {
		runtime.Gosched()
	}
	x = 1
}

func TestNoRaceAtomicStoreCASInt32(t *testing.T) {
	var x int64
	var s int32
	go func() {
		x = 2
		atomic.StoreInt32(&s, 1)
	}()
	for !atomic.CompareAndSwapInt32(&s, 1, 0) {
		runtime.Gosched()
	}
	x = 1
}

func TestNoRaceAtomicCASLoadInt32(t *testing.T) {
	var x int64
	var s int32
	go func() {
		x = 2
		if !atomic.CompareAndSwapInt32(&s, 0, 1) {
			panic("")
		}
	}()
	for atomic.LoadInt32(&s) != 1 {
		runtime.Gosched()
	}
	x = 1
}

func TestNoRaceAtomicCASCASInt32(t *testing.T) {
	var x int64
	var s int32
	go func() {
		x = 2
		if !atomic.CompareAndSwapInt32(&s, 0, 1) {
			panic("")
		}
	}()
	for !atomic.CompareAndSwapInt32(&s, 1, 0) {
		runtime.Gosched()
	}
	x = 1
}

func TestNoRaceAtomicCASCASInt32_2(t *testing.T) {
	var x1, x2 int8
	var s int32
	ch := make(chan bool, 2)
	go func() {
		x1 = 1
		if !atomic.CompareAndSwapInt32(&s, 0, 1) {
			x2 = 1
		}
		ch <- true
	}()
	go func() {
		x2 = 1
		if !atomic.CompareAndSwapInt32(&s, 0, 1) {
			x1 = 1
		}
		ch <- true
	}()
	<-ch
	<-ch
}

func TestNoRaceAtomicLoadInt64(t *testing.T) {
	var x int32
	var s int64
	go func() {
		x = 2
		atomic.AddInt64(&s, 1)
	}()
	for atomic.LoadInt64(&s) != 1 {
		runtime.Gosched()
	}
	x = 1
}

func TestNoRaceAtomicCASCASUInt64(t *testing.T) {
	var x int64
	var s uint64
	go func() {
		x = 2
		if !atomic.CompareAndSwapUint64(&s, 0, 1) {
			panic("")
		}
	}()
	for !atomic.CompareAndSwapUint64(&s, 1, 0) {
		runtime.Gosched()
	}
	x = 1
}

func TestNoRaceAtomicLoadStorePointer(t *testing.T) {
	var x int64
	var s unsafe.Pointer
	var y int = 2
	var p unsafe.Pointer = unsafe.Pointer(&y)
	go func() {
		x = 2
		atomic.StorePointer(&s, p)
	}()
	for atomic.LoadPointer(&s) != p {
		runtime.Gosched()
	}
	x = 1
}

func TestNoRaceAtomicStoreCASUint64(t *testing.T) {
	var x int64
	var s uint64
	go func() {
		x = 2
		atomic.StoreUint64(&s, 1)
	}()
	for !atomic.CompareAndSwapUint64(&s, 1, 0) {
		runtime.Gosched()
	}
	x = 1
}

// Races with non-atomic loads are not detected.
func TestRaceFailingAtomicStoreLoad(t *testing.T) {
	c := make(chan bool)
	var a uint64
	go func() {
		atomic.StoreUint64(&a, 1)
		c <- true
	}()
	_ = a
	<-c
}

func TestRaceAtomicLoadStore(t *testing.T) {
	c := make(chan bool)
	var a uint64
	go func() {
		_ = atomic.LoadUint64(&a)
		c <- true
	}()
	a = 1
	<-c
}

// Races with non-atomic loads are not detected.
func TestRaceFailingAtomicAddLoad(t *testing.T) {
	c := make(chan bool)
	var a uint64
	go func() {
		atomic.AddUint64(&a, 1)
		c <- true
	}()
	_ = a
	<-c
}

func TestRaceAtomicAddStore(t *testing.T) {
	c := make(chan bool)
	var a uint64
	go func() {
		atomic.AddUint64(&a, 1)
		c <- true
	}()
	a = 42
	<-c
}

// A nil pointer in an atomic operation should not deadlock
// the rest of the program. Used to hang indefinitely.
func TestNoRaceAtomicCrash(t *testing.T) {
	var mutex sync.Mutex
	var nilptr *int32
	panics := 0
	defer func() {
		if x := recover(); x != nil {
			mutex.Lock()
			panics++
			mutex.Unlock()
		} else {
			panic("no panic")
		}
	}()
	atomic.AddInt32(nilptr, 1)
}
