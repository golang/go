// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This program produced false race reports when run under the C/C++
// ThreadSanitizer, as it did not understand the synchronization in
// the Go code.

/*
#cgo CFLAGS: -fsanitize=thread
#cgo LDFLAGS: -fsanitize=thread

int val;

int getVal() {
	return val;
}

void setVal(int i) {
	val = i;
}
*/
import "C"

import (
	"runtime"
)

func main() {
	runtime.LockOSThread()
	C.setVal(1)
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		C.setVal(2)
		c <- true
	}()
	<-c
	if v := C.getVal(); v != 2 {
		panic(v)
	}
}
