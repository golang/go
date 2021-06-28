// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// Test that SIGPROF received in C code does not crash the process
// looking for the C code's func pointer.

// The test fails when the function is the first C function.
// The exported functions are the first C functions, so we use that.

// extern void CallGoNop();
import "C"

import (
	"bytes"
	"fmt"
	"runtime/pprof"
	"time"
)

func init() {
	register("CgoCCodeSIGPROF", CgoCCodeSIGPROF)
}

//export GoNop
func GoNop() {}

func CgoCCodeSIGPROF() {
	c := make(chan bool)
	go func() {
		<-c
		start := time.Now()
		for i := 0; i < 1e7; i++ {
			if i%1000 == 0 {
				if time.Since(start) > time.Second {
					break
				}
			}
			C.CallGoNop()
		}
		c <- true
	}()

	var buf bytes.Buffer
	pprof.StartCPUProfile(&buf)
	c <- true
	<-c
	pprof.StopCPUProfile()

	fmt.Println("OK")
}
