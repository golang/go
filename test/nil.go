// $G $F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"time"
)

type T struct {
	i int
}

type IN interface{}

func main() {
	var i *int
	var f *float32
	var s *string
	var m map[float32]*int
	var c chan int
	var t *T
	var in IN
	var ta []IN

	i = nil
	f = nil
	s = nil
	m = nil
	c = nil
	t = nil
	i = nil
	ta = make([]IN, 1)
	ta[0] = nil

	_, _, _, _, _, _, _, _ = i, f, s, m, c, t, in, ta

	arraytest()
	chantest()
	maptest()
	slicetest()
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("not panicking")
		}
	}()
	f()
}

func shouldBlock(f func()) {
	go func() {
		f()
		panic("did not block")
	}()
	time.Sleep(1e7)
}

// nil array pointer

func arraytest() {
	var p *[10]int

	// Looping over indices is fine.
	s := 0
	for i := range p {
		s += i
	}
	if s != 45 {
		panic(s)
	}

	s = 0
	for i := 0; i < len(p); i++ {
		s += i
	}
	if s != 45 {
		panic(s)
	}

	// Looping over values is not.
	shouldPanic(func() {
		for i, v := range p {
			s += i + v
		}
	})

	shouldPanic(func() {
		for i := 0; i < len(p); i++ {
			s += p[i]
		}
	})
}

// nil channel
// select tests already handle select on nil channel

func chantest() {
	var ch chan int

	// nil channel is never ready
	shouldBlock(func() {
		ch <- 1
	})
	shouldBlock(func() {
		<-ch
	})
	shouldBlock(func() {
		x, ok := <-ch
		println(x, ok)
	})

	if len(ch) != 0 {
		panic(len(ch))
	}
	if cap(ch) != 0 {
		panic(cap(ch))
	}
}

// nil map

func maptest() {
	var m map[int]int

	// nil map appears empty
	if len(m) != 0 {
		panic(len(m))
	}
	if m[1] != 0 {
		panic(m[1])
	}
	if x, ok := m[1]; x != 0 || ok {
		panic(fmt.Sprint(x, ok))
	}

	for k, v := range m {
		panic(k)
		panic(v)
	}

	// but cannot be written to
	shouldPanic(func() {
		m[2] = 3
	})
	shouldPanic(func() {
		m[2] = 0, false
	})
}

// nil slice

func slicetest() {
	var x []int

	// nil slice is just a 0-element slice.
	if len(x) != 0 {
		panic(len(x))
	}
	if cap(x) != 0 {
		panic(cap(x))
	}

	// no 0-element slices can be read from or written to
	var s int
	shouldPanic(func() {
		s += x[1]
	})
	shouldPanic(func() {
		x[2] = s
	})
}
