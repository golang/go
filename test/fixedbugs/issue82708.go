// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 82708: the make+copy optimization must not evaluate the
// copy source before the make has checked its length.

package main

import (
	"fmt"
	"strings"
)

var calls int
var sink []int

//go:noinline
func src() []int {
	calls++
	return []int{1, 2, 3}
}

//go:noinline
func nilsrc() []int {
	var p *int
	return []int{*p}
}

func expectMakePanic(name string, f func()) {
	defer func() {
		r := recover()
		if r == nil {
			panic(name + ": no panic")
		}
		if s := fmt.Sprint(r); !strings.Contains(s, "makeslice: len out of range") {
			panic(name + ": wrong panic: " + s)
		}
	}()
	f()
}

func main() {
	n := -1

	expectMakePanic("call", func() {
		m := make([]int, n)
		copy(m, src())
		sink = m
	})
	if calls != 0 {
		panic(fmt.Sprintf("src called %d times, want 0", calls))
	}

	expectMakePanic("nil deref", func() {
		m := make([]int, n)
		copy(m, nilsrc())
		sink = m
	})

	expectMakePanic("slice expr", func() {
		var s []int
		m := make([]int, n)
		copy(m, s[10:])
		sink = m
	})

	expectMakePanic("deref", func() {
		var p *[]int
		m := make([]int, n)
		copy(m, *p)
		sink = m
	})

	expectMakePanic("receive", func() {
		ch := make(chan []int, 1)
		ch <- []int{1}
		defer func() {
			if len(ch) != 1 {
				panic("value received from channel")
			}
		}()
		m := make([]int, n)
		copy(m, <-ch)
		sink = m
	})
}
