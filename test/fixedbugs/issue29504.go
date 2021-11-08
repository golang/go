// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure that in code involving indexing, the bounds
// check always fails at the line number of the '[' token.

package main

import (
	"fmt"
	"runtime"
	"strings"
)

type T struct{ a, b, c, d, e int } // unSSAable

func main() {
	shouldPanic(func() {
		var a [1]int
		sink = a /*line :999999:1*/ [ /*line :100:1*/ i]
	})
	shouldPanic(func() {
		var a [3]int
		sink = a /*line :999999:1*/ [ /*line :200:1*/ i]
	})
	shouldPanic(func() {
		var a []int
		sink = a /*line :999999:1*/ [ /*line :300:1*/ i]
	})
	shouldPanic(func() {
		var a [1]int
		a /*line :999999:1*/ [ /*line :400:1*/ i] = 1
	})
	shouldPanic(func() {
		var a [3]int
		a /*line :999999:1*/ [ /*line :500:1*/ i] = 1
	})
	shouldPanic(func() {
		var a []int
		a /*line :999999:1*/ [ /*line :600:1*/ i] = 1
	})

	shouldPanic(func() {
		var a [3]T
		sinkT = a /*line :999999:1*/ [ /*line :700:1*/ i]
	})
	shouldPanic(func() {
		var a []T
		sinkT = a /*line :999999:1*/ [ /*line :800:1*/ i]
	})
	shouldPanic(func() {
		var a [3]T
		a /*line :999999:1*/ [ /*line :900:1*/ i] = T{}
	})
	shouldPanic(func() {
		var a []T
		a /*line :999999:1*/ [ /*line :1000:1*/ i] = T{}
	})

	shouldPanic(func() {
		var a [3]int
		sinkS = a /*line :999999:1*/ [ /*line :1100:1*/ i:]
	})
	shouldPanic(func() {
		var a []int
		sinkS = a /*line :999999:1*/ [ /*line :1200:1*/ i:]
	})
	shouldPanic(func() {
		var a [3]int
		sinkS = a /*line :999999:1*/ [: /*line :1300:1*/ i]
	})
	shouldPanic(func() {
		var a []int
		sinkS = a /*line :999999:1*/ [: /*line :1400:1*/ i]
	})

	shouldPanic(func() {
		var a [3]T
		sinkST = a /*line :999999:1*/ [ /*line :1500:1*/ i:]
	})
	shouldPanic(func() {
		var a []T
		sinkST = a /*line :999999:1*/ [ /*line :1600:1*/ i:]
	})
	shouldPanic(func() {
		var a [3]T
		sinkST = a /*line :999999:1*/ [: /*line :1700:1*/ i]
	})
	shouldPanic(func() {
		var a []T
		sinkST = a /*line :999999:1*/ [: /*line :1800:1*/ i]
	})

	shouldPanic(func() {
		s := "foo"
		sinkB = s /*line :999999:1*/ [ /*line :1900:1*/ i]
	})
	shouldPanic(func() {
		s := "foo"
		sinkStr = s /*line :999999:1*/ [ /*line :2000:1*/ i:]
	})
	shouldPanic(func() {
		s := "foo"
		sinkStr = s /*line :999999:1*/ [: /*line :2100:1*/ i]
	})

	if bad {
		panic("ERRORS")
	}
}

var i = 9
var sink int
var sinkS []int
var sinkT T
var sinkST []T
var sinkB byte
var sinkStr string

var bad = false

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("did not panic")
		}
		var pcs [10]uintptr
		n := runtime.Callers(1, pcs[:])
		iter := runtime.CallersFrames(pcs[:n])
		buf := ""
		for {
			frame, more := iter.Next()
			buf += fmt.Sprintf("%s:%d %s\n", frame.File, frame.Line, frame.Function)
			if !more {
				break
			}
		}
		if !strings.Contains(buf, "999999") {
			fmt.Printf("could not find marker line in traceback:\n%s\n", buf)
			bad = true
		}
	}()
	f()
}
