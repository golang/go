// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test to make sure we don't think values are dead
// when they are assigned to a PPARAMOUT slot before
// the last GC safepoint.

package main

import (
	"fmt"
	"runtime"
)

// When a T is deallocated, T[1] is certain to
// get clobbered (the runtime writes 0xdeaddeaddeaddead there).
type T [4]int

func f() (r, s *T) {
	r = &T{0x30, 0x31, 0x32, 0x33}
	runtime.GC()
	s = &T{0x40, 0x41, 0x42, 0x43}
	runtime.GC()
	return
}

func main() {
	r, s := f()
	if r[1] != 0x31 {
		fmt.Printf("bad r[1], want 0x31 got %x\n", r[1])
	}
	if s[1] != 0x41 {
		fmt.Printf("bad s[1], want 0x41 got %x\n", s[1])
	}
}
