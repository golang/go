// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	s   [1]string
	pad [16]uintptr
}

//go:noinline
func f(t *int, p *int) []T {
	var res []T
	for {
		var e *T
		res = append(res, *e)
	}
}

func main() {
	defer func() {
		useStack(100) // force a stack copy
		// We're expecting a panic.
		// The bug in this issue causes a throw, which this recover() will not squash.
		recover()
	}()
	junk() // fill the stack with invalid pointers
	f(nil, nil)
}

func useStack(n int) {
	if n == 0 {
		return
	}
	useStack(n - 1)
}

//go:noinline
func junk() uintptr {
	var a [128]uintptr // 1k of bad pointers on the stack
	for i := range a {
		a[i] = 0xaa
	}
	return a[12]
}
