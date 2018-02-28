// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll_test

import (
	. "internal/poll"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

func TestMutexLock(t *testing.T) {
	var mu FDMutex

	if !mu.Incref() {
		t.Fatal("broken")
	}
	if mu.Decref() {
		t.Fatal("broken")
	}

	if !mu.RWLock(true) {
		t.Fatal("broken")
	}
	if mu.RWUnlock(true) {
		t.Fatal("broken")
	}

	if !mu.RWLock(false) {
		t.Fatal("broken")
	}
	if mu.RWUnlock(false) {
		t.Fatal("broken")
	}
}

func TestMutexClose(t *testing.T) {
	var mu FDMutex
	if !mu.IncrefAndClose() {
		t.Fatal("broken")
	}

	if mu.Incref() {
		t.Fatal("broken")
	}
	if mu.RWLock(true) {
		t.Fatal("broken")
	}
	if mu.RWLock(false) {
		t.Fatal("broken")
	}
	if mu.IncrefAndClose() {
		t.Fatal("broken")
	}
}

func TestMutexCloseUnblock(t *testing.T) {
	c := make(chan bool)
	var mu FDMutex
	mu.RWLock(true)
	for i := 0; i < 4; i++ {
		go func() {
			if mu.RWLock(true) {
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
	mu.IncrefAndClose() // Must unblock the readers.
	for i := 0; i < 4; i++ {
		select {
		case <-c:
		case <-time.After(10 * time.Second):
			t.Fatal("broken")
		}
	}
	if mu.Decref() {
		t.Fatal("broken")
	}
	if !mu.RWUnlock(true) {
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

	var mu FDMutex
	ensurePanics(func() { mu.Decref() })
	ensurePanics(func() { mu.RWUnlock(true) })
	ensurePanics(func() { mu.RWUnlock(false) })

	ensurePanics(func() { mu.Incref(); mu.Decref(); mu.Decref() })
	ensurePanics(func() { mu.RWLock(true); mu.RWUnlock(true); mu.RWUnlock(true) })
	ensurePanics(func() { mu.RWLock(false); mu.RWUnlock(false); mu.RWUnlock(false) })

	// ensure that it's still not broken
	mu.Incref()
	mu.Decref()
	mu.RWLock(true)
	mu.RWUnlock(true)
	mu.RWLock(false)
	mu.RWUnlock(false)
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
	var mu FDMutex
	var readState [2]uint64
	var writeState [2]uint64
	for p := 0; p < P; p++ {
		go func() {
			r := rand.New(rand.NewSource(rand.Int63()))
			for i := 0; i < N; i++ {
				switch r.Intn(3) {
				case 0:
					if !mu.Incref() {
						t.Error("broken")
						return
					}
					if mu.Decref() {
						t.Error("broken")
						return
					}
				case 1:
					if !mu.RWLock(true) {
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
					if mu.RWUnlock(true) {
						t.Error("broken")
						return
					}
				case 2:
					if !mu.RWLock(false) {
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
					if mu.RWUnlock(false) {
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
	if !mu.IncrefAndClose() {
		t.Fatal("broken")
	}
	if !mu.Decref() {
		t.Fatal("broken")
	}
}
