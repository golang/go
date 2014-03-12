// run

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that s[len(s):] - which can point past the end of the allocated block -
// does not confuse the garbage collector.

package main

import (
	"runtime"
	"time"
)

type T struct {
	ptr **int
	pad [120]byte
}

var things []interface{}

func main() {
	setup()
	runtime.GC()
	runtime.GC()
	time.Sleep(10*time.Millisecond)
	runtime.GC()
	runtime.GC()
	time.Sleep(10*time.Millisecond)
}

func setup() {
	var Ts []interface{}
	buf := make([]byte, 128)
	
	for i := 0; i < 10000; i++ {
		s := string(buf)
		t := &T{ptr: new(*int)}
		runtime.SetFinalizer(t.ptr, func(**int) { panic("*int freed too early") })
		Ts = append(Ts, t)
		things = append(things, s[len(s):])
	}
	
	things = append(things, Ts...)
}

