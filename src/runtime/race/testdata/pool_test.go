// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"sync"
	"testing"
	"time"
)

func TestRacePool(t *testing.T) {
	// Pool randomly drops the argument on the floor during Put.
	// Repeat so that at least one iteration gets reuse.
	for i := 0; i < 10; i++ {
		c := make(chan int)
		p := &sync.Pool{New: func() interface{} { return make([]byte, 10) }}
		x := p.Get().([]byte)
		x[0] = 1
		p.Put(x)
		go func() {
			y := p.Get().([]byte)
			y[0] = 2
			c <- 1
		}()
		x[0] = 3
		<-c
	}
}

func TestNoRacePool(t *testing.T) {
	for i := 0; i < 10; i++ {
		p := &sync.Pool{New: func() interface{} { return make([]byte, 10) }}
		x := p.Get().([]byte)
		x[0] = 1
		p.Put(x)
		go func() {
			y := p.Get().([]byte)
			y[0] = 2
			p.Put(y)
		}()
		time.Sleep(100 * time.Millisecond)
		x = p.Get().([]byte)
		x[0] = 3
	}
}
