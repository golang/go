// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sync_test

import (
	"runtime"
	. "sync"
	"sync/atomic"
	"testing"
)

type one int

func (o *one) Increment() {
	*o++
}

func run(t *testing.T, once *Once, o *one, c chan bool) {
	once.Do(func() { o.Increment() })
	if v := *o; v != 1 {
		t.Errorf("once failed inside run: %d is not 1", v)
	}
	c <- true
}

func TestOnce(t *testing.T) {
	o := new(one)
	once := new(Once)
	c := make(chan bool)
	const N = 10
	for i := 0; i < N; i++ {
		go run(t, once, o, c)
	}
	for i := 0; i < N; i++ {
		<-c
	}
	if *o != 1 {
		t.Errorf("once failed outside run: %d is not 1", *o)
	}
}

func TestOncePanic(t *testing.T) {
	once := new(Once)
	for i := 0; i < 2; i++ {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("Once.Do() has not panic'ed")
				}
			}()
			once.Do(func() {
				panic("failed")
			})
		}()
	}
	once.Do(func() {})
	once.Do(func() {
		t.Fatalf("Once called twice")
	})
}

func BenchmarkOnce(b *testing.B) {
	const CallsPerSched = 1000
	procs := runtime.GOMAXPROCS(-1)
	N := int32(b.N / CallsPerSched)
	var once Once
	f := func() {}
	c := make(chan bool, procs)
	for p := 0; p < procs; p++ {
		go func() {
			for atomic.AddInt32(&N, -1) >= 0 {
				runtime.Gosched()
				for g := 0; g < CallsPerSched; g++ {
					once.Do(f)
				}
			}
			c <- true
		}()
	}
	for p := 0; p < procs; p++ {
		<-c
	}
}
