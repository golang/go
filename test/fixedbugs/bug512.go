// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Gccgo did not make a copy of a value receiver when using a
// goroutine to call a method.

package main

import (
	"sync"
	"sync/atomic"
)

var wg sync.WaitGroup

type S struct {
	i1, i2 int32
}

var done int32

func (s S) Check(v1, v2 int32) {
	for {
		if g1 := atomic.LoadInt32(&s.i1); v1 != g1 {
			panic(g1)
		}
		if g2 := atomic.LoadInt32(&s.i2); v2 != g2 {
			panic(g2)
		}
		if atomic.LoadInt32(&done) != 0 {
			break
		}
	}
	wg.Done()
}

func F() {
	s := S{1, 2}
	go s.Check(1, 2)
	atomic.StoreInt32(&s.i1, 3)
	atomic.StoreInt32(&s.i2, 4)
	atomic.StoreInt32(&done, 1)
}

func main() {
	wg.Add(1)
	F()
	wg.Wait()
}
