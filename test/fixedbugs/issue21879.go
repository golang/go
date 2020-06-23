// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
)

func main() {
	println(caller().frame.Function)

	// Used to erroneously print "main.call.name" instead of
	// "main.main".
	println(caller().name())
}

func caller() call {
	var pcs [3]uintptr
	n := runtime.Callers(1, pcs[:])
	frames := runtime.CallersFrames(pcs[:n])
	frame, _ := frames.Next()
	frame, _ = frames.Next()

	return call{frame: frame}
}

type call struct {
	frame runtime.Frame
}

func (c call) name() string {
	return c.frame.Function
}
