// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test that SIGPROF received in C code does not crash the process
// looking for the C code's func pointer.

// The test fails when the function is the first C function.
// The exported functions are the first C functions, so we use that.

// extern void GoNop();
import "C"

import (
	"bytes"
	"fmt"
	"runtime/pprof"
)

func init() {
	register("CgoCCodeSIGPROF", CgoCCodeSIGPROF)
}

//export GoNop
func GoNop() {}

func CgoCCodeSIGPROF() {
	c := make(chan bool)
	go func() {
		for {
			<-c
			for i := 0; i < 1e7; i++ {
				C.GoNop()
			}
			c <- true
		}
	}()

	var buf bytes.Buffer
	pprof.StartCPUProfile(&buf)
	c <- true
	<-c
	pprof.StopCPUProfile()

	fmt.Println("OK")
}
