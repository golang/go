// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"sync"
	"time"
)

type B struct {
	pid int
	f   func() (uint64, error)
	wg  sync.WaitGroup
	v   uint64
}

func newB(pid int) *B {
	return &B{
		pid: pid,
	}
}

//go:noinline
func Sq(i int) uint64 {
	S++
	return uint64(i * i)
}

type RO func(*B)

var ROSL = []RO{
	Bad(),
}

func Bad() RO {
	return func(b *B) {
		b.f = func() (uint64, error) {
			return Sq(b.pid), nil
		}
	}
}

func (b *B) startit() chan<- struct{} {
	stop := make(chan struct{})
	b.wg.Add(1)
	go func() {
		defer b.wg.Done()
		var v uint64
		for {
			select {
			case <-stop:
				b.v = v
				return
			case <-time.After(1 * time.Millisecond):
				r, err := b.f()
				if err != nil {
					panic("bad")
				}
				v = r
			}
		}
	}()
	return stop
}

var S, G int

//go:noinline
func rec(x int) int {
	if x == 0 {
		return 9
	}
	return rec(x-1) + 1
}

//go:noinline
func recur(x int) {
	for i := 0; i < x; i++ {
		G = rec(i)
	}
}

func main() {
	b := newB(17)
	for _, opt := range ROSL {
		opt(b)
	}
	stop := b.startit()

	// see if we can get some stack growth/moving
	recur(10101)

	if stop != nil {
		stop <- struct{}{}
	}
}
