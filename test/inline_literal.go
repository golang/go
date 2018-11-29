// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"log"
	"reflect"
	"runtime"
)

func hello() string {
	return "Hello World" // line 16
}

func foo() string { // line 19
	x := hello() // line 20
	y := hello() // line 21
	return x + y // line 22
}

func bar() string {
	x := hello() // line 26
	return x
}

// funcPC returns the PC for the func value f.
func funcPC(f interface{}) uintptr {
	return reflect.ValueOf(f).Pointer()
}

// Test for issue #15453. Previously, line 26 would appear in foo().
func main() {
	pc := funcPC(foo)
	f := runtime.FuncForPC(pc)
	for ; runtime.FuncForPC(pc) == f; pc++ {
		file, line := f.FileLine(pc)
		if line == 0 {
			continue
		}
		// Line 16 can appear inside foo() because PC-line table has
		// innermost line numbers after inlining.
		if line != 16 && !(line >= 19 && line <= 22) {
			log.Fatalf("unexpected line at PC=%d: %s:%d\n", pc, file, line)
		}
	}
}
