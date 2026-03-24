// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: disable test for ppc64le/dynlink? See cmd/compile/internal/reader/noder.go:addTailCall.

package main

import (
	"fmt"
	"runtime"
	"time"
)

type I interface {
	foo() time.Duration // foo returns its running time
}

type base struct {
}

func (b *base) foo() time.Duration {
	t := time.Now()
	var pcs [10]uintptr
	runtime.Callers(1, pcs[:])
	return time.Since(t)
}

type wrap struct {
	I
	data int
}

// best runs f a bunch of times, picks the shortest returned duration.
func best(f func() time.Duration) time.Duration {
	m := f()
	for range 9 {
		m = min(m, f())
	}
	return m
}

func main() {
	if runtime.GOARCH == "wasm" {
		// TODO: upgrade wasm to do indirect tail calls
		return
	}
	var i I = &base{}
	for x := range 1000 {
		i = &wrap{I: i, data: x}
	}
	short := best(i.foo)
	for x := range 9000 {
		i = &wrap{I: i, data: x}
	}
	long := best(i.foo)

	ratio := long.Seconds() / short.Seconds()

	// Running time should be independent of the number of wrappers.
	// Prior to the fix for 75764, it was linear in the number of wrappers.
	// Pre-fix, we get ratios typically in the 7.0-10.0 range.
	// Post-fix, it is in the 1.0-1.5 range.
	allowed := 5.0
	if ratio >= allowed {
		fmt.Printf("short: %v\nlong: %v\nratio: %v\nallowed: %v\n", short, long, ratio, allowed)
	}
}
