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

extern void GoRun(void);

// Yes, you can have definitions if you use //export, as long as they are weak.

int val __attribute__ ((weak));

int run(void) __attribute__ ((weak));

int run() {
	val = 1;
	GoRun();
	return val;
}

void setVal(int) __attribute__ ((weak));

void setVal(int i) {
	val = i;
}
*/
import "C"

import "runtime"

//export GoRun
func GoRun() {
	runtime.LockOSThread()
	c := make(chan bool)
	go func() {
		runtime.LockOSThread()
		C.setVal(2)
		c <- true
	}()
	<-c
}

func main() {
	if v := C.run(); v != 2 {
		panic(v)
	}
}
