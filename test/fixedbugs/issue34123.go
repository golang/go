// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure that the line number is reported correctly
// for faulting instructions.

package main

import (
	"fmt"
	"runtime"
)

var x byte
var p *byte

//go:noinline
func f() {
	q := p
	x = 11  // line 23
	*q = 12 // line 24
}
func main() {
	defer func() {
		recover()
		var pcs [10]uintptr
		n := runtime.Callers(1, pcs[:])
		frames := runtime.CallersFrames(pcs[:n])
		for {
			f, more := frames.Next()
			if f.Function == "main.f" && f.Line != 24 {
				panic(fmt.Errorf("expected line 24, got line %d", f.Line))
			}
			if !more {
				break
			}
		}
	}()
	f()
}
