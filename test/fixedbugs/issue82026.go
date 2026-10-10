// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 82026: CSE of OpLocalAddr across calls lets GC free objects
// still reachable from a local array.

package main

import (
	"fmt"
	"os"
	"runtime"
)

type payload struct{ v [64]int }

type T struct {
	p    *payload
	pad  [8]int // keeps T out of SSA registers
	flag *int
}

var sink []*payload

//go:noinline
func churn() {
	runtime.GC()
	for i := 0; i < 64; i++ {
		sink = append(sink[:0], &payload{})
	}
	runtime.GC()
}

//go:noinline
func newPayload() *payload {
	p := &payload{}
	p.v[7] = 42
	return p
}

//go:noinline
func use(t T) int { return t.p.v[7] }

//go:noinline
func f(a, b T, pick int) int {
	arr := [2]T{a, b}
	m := pick & 1
	if arr[m].p == nil {
		return -1
	}
	churn()
	return use(arr[m])
}

func main() {
	bad := 0
	for i := 0; i < 2000; i++ {
		a := T{p: newPayload()}
		b := T{p: newPayload()}
		if got := f(a, b, i); got != 42 {
			bad++
		}
	}
	if bad != 0 {
		fmt.Printf("FAILED: bad = %d\n", bad)
		os.Exit(1)
	}
}
