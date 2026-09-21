// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 79186: on ppc64le (POWER8/9), atomic add operations lacked a
// post-barrier (acquire ordering), allowing loads after an RWMutex.RLock
// to be speculatively reordered before the lock acquisition, causing
// concurrent map read and map write.

package main

import (
	"runtime"
	"sync"
)

type M struct {
	mu sync.RWMutex
	m  map[int]int
}

func NewM() *M {
	return &M{m: make(map[int]int)}
}

func (x *M) Get(k int) (int, bool) {
	x.mu.RLock()
	v, ok := x.m[k]
	x.mu.RUnlock()
	return v, ok
}

func (x *M) Set(k, v int) {
	x.mu.Lock()
	x.m[k] = v
	x.mu.Unlock()
}

func main() {
	runtime.GOMAXPROCS(2)

	x := NewM()

	const goroutines = 256
	const iters = 200000

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				k := (id + i) & 15
				if _, ok := x.Get(k); !ok {
					x.Set(k, i)
				} else if i&7 == 0 {
					x.Set(k, i)
				}
			}
		}(g)
	}

	wg.Wait()
}
