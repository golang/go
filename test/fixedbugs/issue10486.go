// run

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 10486.
// Check stack walk during div by zero fault,
// especially on software divide systems.

package main

import "runtime"

var A, B int

func divZero() int {
	defer func() {
		if p := recover(); p != nil {
			var pcs [512]uintptr
			runtime.Callers(2, pcs[:])
			runtime.GC()
		}
	}()
	return A / B
}

func main() {
	A = 1
	divZero()
}
