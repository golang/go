// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"math/rand"
	"runtime"
	"testing"
	"time"
)

func TestMutexLock(t *testing.T) {
	var mu fdMutex

	if !mu.incref() {
		t.Fatal("broken")
	}
	if mu.decref() {
		t.Fatal("broken")
	}

	if !mu.rwlock(true) {
		t.Fatal("broken")
	}
	if mu.rwunlock(true) {
		t.Fatal("broken")
	}

	if !mu.rwlock(false) {
		t.Fatal("broken")
	}
	if mu.rwunlock(false) {
		t.Fatal("broken")
	}
}

func TestMutexClose(t *testing.T) {
	var mu fdMutex
	if !mu.increfAndClose() {
		t.Fatal("broken")
	}

	if mu.incref() {
		t.Fatal("broken")
	}
	if mu.rwlock(true) {
		t.Fatal("broken")
	}
	if mu.rwlock(false) {
		t.Fatal("broken")
	}
	if mu.increfAndClose() {
		t.Fatal("broken")
	}
}

func TestMutexCloseUnblock(t *testing.T) {
	c := make(chan bool)
	var mu fdMutex
	mu.rwlock(true)
	for i := 0; i < 4; i++ {
		go func() {
			if mu.rwlock(true) {
				t.Error("broken")
				return
			}
			c <- true
		}()
	}
	// Concurrent goroutines must not be able to read lock the mutex.
	time.Sleep(time.Millisecond)
	select {
	case <-c:
		t.Fatal("broken")
	default:
	}
	mu.increfAndClose() // Must unblock the readers.
	for i := 0; i < 4; i++ {
		select {
		case <-c:
		case <-time.After(10 * time.Second):
			t.Fatal("broken")
		}
	}
	if mu.decref() {
		t.Fatal("broken")
	}
	if !mu.rwunlock(true) {
		t.Fatal("broken")
	}
}

func TestMutexPanic(t *testing.T) {
	ensurePanics := func(f func()) {
		defer func() {
			if recover() == nil {
				t.Fatal("does not panic")
			}
		}()
		f()
	}

	var mu fdMutex
	ensurePanics(func() { mu.decref() })
	ensurePanics(func() { mu.rwunlock(true) })
	ensurePanics(func() { mu.rwunlock(false) })

	ensurePanics(func() { mu.incref(); mu.decref(); mu.decref() })
	ensurePanics(func() { mu.rwlock(true); mu.rwunlock(true); mu.rwunlock(true) })
	ensurePanics(func() { mu.rwlock(false); mu.rwunlock(false); mu.rwunlock(false) })

	// ensure that it's still not broken
	mu.incref()
	mu.decref()
	mu.rwlock(true)
	mu.rwunlock(true)
	mu.rwlock(false)
	mu.rwunlock(false)
}

func TestMutexStress(t *testing.T) {
	P := 8
	N := int(1e6)
	if testing.Short() {
		P = 4
		N = 1e4
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(P))
	done := make(chan bool)
	var mu fdMutex
	var readState [2]uint64
	var writeState [2]uint64
	for p := 0; p < P; p++ {
		go func() {
			r := rand.New(rand.NewSource(rand.Int63()))
			for i := 0; i < N; i++ {
				switch r.Intn(3) {
				case 0:
					if !mu.incref() {
						t.Error("broken")
						return
					}
					if mu.decref() {
						t.Error("broken")
						return
					}
				case 1:
					if !mu.rwlock(true) {
						t.Error("broken")
						return
					}
					// Ensure that it provides mutual exclusion for readers.
					if readState[0] != readState[1] {
						t.Error("broken")
						return
					}
					readState[0]++
					readState[1]++
					if mu.rwunlock(true) {
						t.Error("broken")
						return
					}
				case 2:
					if !mu.rwlock(false) {
						t.Error("broken")
						return
					}
					// Ensure that it provides mutual exclusion for writers.
					if writeState[0] != writeState[1] {
						t.Error("broken")
						return
					}
					writeState[0]++
					writeState[1]++
					if mu.rwunlock(false) {
						t.Error("broken")
						return
					}
				}
			}
			done <- true
		}()
	}
	for p := 0; p < P; p++ {
		<-done
	}
	if !mu.increfAndClose() {
		t.Fatal("broken")
	}
	if !mu.decref() {
		t.Fatal("broken")
	}
}
